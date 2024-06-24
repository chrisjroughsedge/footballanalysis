from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
import json

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        self.camera_angles = ["Left", "Middle", "Right"]
        self.complete_passes = []

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
        
        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        # commented out for distance changes
        # overlay = frame.copy()
        # cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        # alpha = 0.4
        # cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # # Get the number of time each team had ball control
        # team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        # team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        # team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        # team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        # cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        # cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"FrameNumber: {frame_num}",(1400,1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        ball_positions = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():

                ########## start my code ##############
                for test, ball in ball_dict.items():
                    ball_pos = ball["bbox"]

                #if frame_num == 136:
                playerDict = {
                    "playerLocation": player["bbox"],
                    "track_id": int(track_id),
                    "frame_num": int(frame_num),
                    "assignedPlayer": int(0),
                    "ball_position": ball_pos,
                    "playerTeam": int(tracks['players'][frame_num][track_id]['team']),
                    "footPosition": tracks['players'][frame_num][track_id]['position']
                    # "positionTransformed": player['position_transformed'],
                    # "positionAdjusted": player['position_adjusted']
                }
                
                

                ########## end my code ##############

                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball', False):
                    #if frame_num == 136:
                    playerDict['assignedPlayer'] = int(track_id)
                    
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

                #if frame_num == 136:
                ball_positions.append(playerDict)

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)
        
        self.drawPasses(ball_positions)
        return output_video_frames
    
    def drawPasses(self, ball_positions):
        # with open("data/data.json", "w") as outfile:
        #     json.dump(ball_positions, outfile, indent=2)
        
        f= open('data/touches.json')
        touchData = json.load(f)
        passes = []
        for touch in touchData:
            get_frame = touch * 25
            filtered_list = [d for d in ball_positions if d['frame_num'] == get_frame]
            if len(filtered_list) > 1:
                passcomplete = {
                    # "startLocation": "",
                    "endLocation": "",
                    "framenum": get_frame,
                    "oldplayer": "",
                    "newplayer": "",
                    "allPlayers": filtered_list
                }
                passes.append(passcomplete)


        camera_angle = ""
        for b in passes:
            playerPoints = []
            ballPositionX = b['allPlayers'][0]['ball_position']
            self.complete_passes.append(ballPositionX)
            passStart = ""
            passEnd = ""
            frame_num = b['allPlayers'][0]['frame_num']
            for players in b['allPlayers']:
                playerPointCameraAngles = []

                for camera_angle in self.camera_angles:
                    playerPoint_angle = self.returnPoints(camera_angle,players['playerLocation'][2], players['playerLocation'][3])
                    playerPointCameraAngles.append(playerPoint_angle)

                playerInformation = {
                    "playerPoint": playerPointCameraAngles, #self.returnPoints(camera_angle,players['playerLocation'][2], players['playerLocation'][3]),
                    "playerTeam": players['playerTeam'],
                    "playerId": players["track_id"]
                }
                playerPoints.append(playerInformation)
        #     # passStart = self.returnPoints(camera_angle, passStart[2], passStart[3])
        #     # passEnd = self.returnPoints(camera_angle, passEnd[2], passEnd[3])

            self.drawImages(playerPoints, passStart, passEnd, frame_num, ballPositionX)
        
    
    # def drawPasses(self, ball_positions):
    #     with open("data/data.json", "w") as outfile:
    #         json.dump(ball_positions, outfile, indent=2)
        
    #     f= open('data/touches.json')
    #     frameData = json.load(f)
    #     frames_to_remove = frameData

    #     passes = []
    #     filtered_list = [d for d in ball_positions if d['assignedPlayer'] != 0]
    #     currentPlayer = filtered_list[0]['assignedPlayer']
    #     currentPlayerLocation = filtered_list[0]['playerLocation']
    #     for x in filtered_list:
    #         if currentPlayer != x['assignedPlayer']:
    #             endLocation = x['playerLocation']
    #             allPlayersInFrame = [d for d in ball_positions if d['frame_num'] == x['frame_num']]
    #             passcomplete = {
    #                 "startLocation": currentPlayerLocation,
    #                 "endLocation": endLocation,
    #                 "framenum": x['frame_num'],
    #                 "oldplayer": currentPlayer,
    #                 "newplayer": x['assignedPlayer'],
    #                 "allPlayers": allPlayersInFrame
    #             }
                
    #             currentPlayerLocation = ""
    #             currentPlayer = x['assignedPlayer']
    #         if currentPlayer == x['assignedPlayer']:
    #             currentPlayerLocation = x['playerLocation']

    #     # camera_angle = ""
    #     for b in passes:
    #         playerPoints = []
    #         ballPositionX = b['allPlayers'][0]['ball_position'][0]
    #         passStart = b['startLocation']
    #         passEnd = b['endLocation']
    #         frame_num = b['allPlayers'][0]['frame_num']
    #         for players in b['allPlayers']:
    #             playerPointCameraAngles = []

    #             for camera_angle in self.camera_angles:
    #                 playerPoint_angle = self.returnPoints(camera_angle,players['playerLocation'][2], players['playerLocation'][3])
    #                 playerPointCameraAngles.append(playerPoint_angle)

    #             playerInformation = {
    #                 "playerPoint": playerPointCameraAngles, #self.returnPoints(camera_angle,players['playerLocation'][2], players['playerLocation'][3]),
    #                 "playerTeam": players['playerTeam'],
    #                 "playerId": players["track_id"]
    #             }
    #             playerPoints.append(playerInformation)
    #         # passStart = self.returnPoints(camera_angle, passStart[2], passStart[3])
    #         # passEnd = self.returnPoints(camera_angle, passEnd[2], passEnd[3])

    #         self.drawImages(playerPoints, passStart, passEnd, frame_num, ballPositionX)
        
    
    def drawImages(self,playerPoints, passStartLocation, passEndLocation, frame_num, ballPositionX):
        # newImg=cv2.imread('input_videos/statsbombpitch.PNG')
        
        #do for all camera angles
        count = 0
        # currentballPosition = ballPositionX
        previousBallLocation = ballPositionX
        print(f'current {ballPositionX}')
        passCount = len(self.complete_passes)
        print(f'passCount: {[passCount]} - pos: {ballPositionX}')
        if passCount > 1:
            for idx in range(1, len(self.complete_passes)):
                print(f'count {passCount - 1}')
                previousBallLocation = self.complete_passes[idx - 1]
                print(f'previous {previousBallLocation}')

        for camera_angle in self.camera_angles:
            newImg=cv2.imread('input_videos/statsbombpitch.PNG')
            cv2.putText(newImg, f'Frame Num: {frame_num}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA) 
            # cv2.putText(newImg, f'Ball Position: {ballPositionX}', (50,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)

            # passStart = self.returnPoints(camera_angle, passStartLocation[2], passStartLocation[3])
            # passEnd = self.returnPoints(camera_angle, passEndLocation[2], passEndLocation[3])
            # cv2.arrowedLine(newImg, passStart, passEnd, (0,255,0), 1)

            currentballPosition = self.returnPoints(camera_angle, ballPositionX[2], ballPositionX[3])
            cv2.circle(newImg, currentballPosition, 10, (0,0,0), -1)

            if passCount > 1:
                print(f'prev abll {previousBallLocation}')
                previousBallPoints = self.returnPoints(camera_angle, previousBallLocation[2], previousBallLocation[3])
                cv2.arrowedLine(newImg, previousBallPoints, currentballPosition, (0,255,0), 1)

            #get previous ball position for arrowed Line
            # passCount = len(self.complete_passes)
            # if passCount >= 1:
            #     previousBallLocation = self.returnPoints(camera_angle, self.complete_passes[passCount - 1][2], self.complete_passes[passCount - 1][3])
            #     print(f'previous {previousBallLocation}')
            #     cv2.arrowedLine(newImg, previousBallLocation, currentballPosition, (0,255,0), 1)
            
                


            for y in playerPoints:
                color = (0,0,0)
                if y['playerTeam'] == 1:
                    color = (0,0,255)
                if y['playerTeam'] == 2:
                    color = (255,0,0)
                cv2.circle(newImg, y['playerPoint'][count] , 4, color, -1)
                
                cv2.putText(newImg, f'{y['playerId']}', y['playerPoint'][count], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)

            cv2.imwrite(f'output_videos/images/Frame_{frame_num}_{camera_angle}.png', newImg)
            count +=1
            
        
        # for y in playerPoints:
        #     color = (0,0,0)
        #     if y['playerTeam'] == 1:
        #         color = (0,0,255)
        #     if y['playerTeam'] == 2:
        #         color = (255,0,0)
        #     cv2.circle(newImg, y['playerPoint'] , 4, color, -1)
            
        #     cv2.putText(newImg, f'{y['playerId']}', y['playerPoint'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)

        # # cv2.rectangle(newImg, (319,670), (356,752), (255,0,0), -1)
        # cv2.imwrite(f'output_videos/images/Frame_{frame_num}.png', newImg)
    
    def getMatrix(self, angle):
        #points of source, actual image
        pts1 = np.float32([[453,269],[1200,257],[1824,849],[63,884]])
        # point of statsbom pitch
        pts2 = np.float32([[330,2],[689,4],[668,631],[318,629]])
        if angle == "Left":
            #points of source, actual image
            pts1 = np.float32([[492,280],[932,289],[646,798],[3,559]])
            # point of statsbom pitch
            pts2 = np.float32([[142,3],[358,3],[338,630],[142,492]])
        if angle == "Middle":
            # pts1 = np.float32([[450,235],[1007,227],[1454,845],[109,887]])
            pts1 = np.float32([[450,235],[1007,227],[1454,1000],[109,1000]])
            pts2 = np.float32([[358,3],[600,3],[587,630],[338,628]])
        if angle == "Right":
            #points of source, actual image
            pts1 = np.float32([[490,317],[1272,317],[1808,502],[870,775]])
            # point of statsbom pitch
            pts2 = np.float32([[600,3],[941,143],[940,491],[587,629]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        matrix=np.array(M)
        return matrix
    
    def returnPoints(self, angle, locationx, locationy):
        pts3 = np.float32([[locationx,locationy]])
        pts3o=cv2.perspectiveTransform(pts3[None, :, :], self.getMatrix(angle))
        x1=int(pts3o[0][0][0])
        y1=int(pts3o[0][0][1])
        pp=(x1,y1)
        # print(pp)
        return pp
        


