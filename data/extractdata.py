import json
f= open('data/data.json')
frameData = json.load(f)
firstFrame = 30*25
lastFrame = 175+25

# filtered_list = [d for d in frameData if d['frame_num'] >= firstFrame and d['frame_num'] < lastFrame]

filtered_list = [d for d in frameData if d['frame_num'] >= firstFrame == firstFrame]

for n in filtered_list:
    print(f' Player: {n['assignedPlayer']}')
