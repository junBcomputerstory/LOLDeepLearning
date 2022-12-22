import numpy as np
import json
from keras.models import load_model
import tensorflow as tf
import os
from matplotlib import pyplot as plt

def create_data(path:str):
    try:
        position_to_num={
            "TOP":0,
            "JUNGLE":1,
            "MIDDLE":2,
            "BOTTOM":3,
            "UTILITY":4
        }
        with open(path,"r+",encoding="utf-8") as f:
            data=json.load(f)
            a=[
                [
                
                    [0]*430 for __ in range(10)
                ] for ___ in range(50)
            ]
            champion_data=data["metadata"]["champions"]
            team_100_win=None
            perk_list=dict()
            for i in range(1,11):
                index=str(i)
                perk_list[index]=champion_data[index]["perk"]
            previous_items_list=[list() for _ in range(5)]
            item_order=[
                list() for _ in range(5)
            ]
            frame_cnt=0
            for frame in data["info"]["frames"]:
                if(frame_cnt>50):
                    break
                frame_data=a[frame_cnt]
                frame_info=frame["participantFrames"]
                for who in frame_info:
                    index=position_to_num[champion_data[who]["position"]]
                    if champion_data[who]["teamId"]==200:
                        team_100_win=(champion_data[who]["win"]==False)
                        index+=5
                    champion_index=champion_data[who]["champion"]
                    item_list=frame_info[who]["items"]
                    frame_data[index][champion_index]=1
                    for i in range(len(item_list)):
                        item_list[i]+=1

                    item_list.sort()
                    frame_data[index][428]=frame_info[who]["currentGold"]
                    frame_data[index][429]=frame_info[who]["totalGold"]
                    for perk in perk_list[who]:
                            frame_data[index][perk]=1
                    for item_id in item_list:
                        frame_data[index][230+item_id]=1
                    if index<5 and previous_items_list[index]!=item_list:
                        for item in item_list:
                            if item not in previous_items_list[index]:
                                item_order[index].append(item)
                        previous_items_list[index]=item_list
                frame_cnt+=1
            data_x=np.array(a[:frame_cnt])
            data_y=[
                        [
                            [0]*230 for _ in range(25)
                        ] for __ in range(5)
                        ]
            for position in range(5):
                for order,item in enumerate(item_order[position]):
                    data_y[position][order][item]=1
            data_y=np.array(data_y)
            if team_100_win:
                data_y=np.array([1,0])
            else:
                data_y=np.array([0,1])
            return data_x,data_y
    except:
        return None,None


if __name__=="__main__":
    #gpus=tf.config.experimental.list_physical_devices("GPU")
    #tf.config.experimental.set_visible_devices(gpus[0],"GPU")
    position_to_num={
        "TOP":0,
        "JUNGLE":1,
        "MIDDLE":2,
        "BOTTOM":3,
        "UTILITY":4
    }
    data_path="./match_data_timeline_include_item_id"  #데이터 경로
    model_path="./backup/7500.h5"   #테스트 모델
    test_data_size=100 #모델 평가를 위한 데이터 개수
    model=load_model(model_path)
    data_set_list=os.listdir(data_path)
    test_size=10000 #테스트 해볼 데이터 개수
    total_size=0
    result=np.array([[0.0]*50]*2)
    for test in data_set_list[20000:20000+test_size]:
        X,Y=create_data(data_path+"/"+test) 
        if X is None:
            continue
        X=X.reshape(X.shape[0],1,X.shape[1]*X.shape[2])
        Y=np.array([Y]*len(X))
        pred_y=model.predict(X)
        pred_y=np.transpose(pred_y.reshape(len(pred_y),2))
        Y=np.transpose(Y.reshape(len(Y),2))
        error=np.abs(Y-pred_y)
        error=np.pad(error,((0,0),(0,50-len(error[0]))), constant_values=0.0)
        result+=error
        total_size+=1
    result/=total_size
    plt.plot(range(50),result[0],color="violet")
    plt.xlabel("1min")
    plt.ylabel("error")
    plt.show()
