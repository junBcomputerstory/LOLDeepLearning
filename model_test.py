import numpy as np
import json
from keras.models import load_model
from keras.utils import Sequence
import os
import random

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

def get_data_set(data_path:str, data_set_list:list,batch_size:int):
    for start in range(0,len(data_set_list),batch_size):
        data_X=np.empty((0,4300))
        data_y=np.empty((0,2))
        for i in range(start,min(start+batch_size,len(data_set_list))):
            test=data_set_list[i]
            X,Y=create_data(data_path+"/"+test)
            if X is None:
                continue
            X=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
            data_X=np.append(data_X,np.array(X),axis=0)
            data_y=np.append(data_y,np.array([Y]*len(X)),axis=0)
        data_X=data_X.reshape(data_X.shape[0],1,data_X.shape[1])
        data_y=data_y.reshape(data_y.shape[0],1,data_y.shape[1])
        yield data_X,data_y


class DataGen(Sequence):
    def __init__(self, path,batch_size,test_size):
        self.path=path
        self.batch_size=batch_size
        self.data_set_list=os.listdir(path)
        random.shuffle(self.data_set_list)
        self.data_set_list=self.data_set_list[:test_size]

    def __len__(self):
        len_=int(len(self.data_set_list)/self.batch_size)
        if len_*self.batch_size<len(self.data_set_list):
            len_+=1
        return len_
    
    def __getitem__(self, index):
        indexes = self.data_set_list[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [k for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X,y

    def __data_generation(self, list_IDs_temp):
        data_X=np.empty((0,4300))
        data_y=np.empty((0,2))
        for ID in list_IDs_temp:
            X,Y=create_data(self.path+"/"+ID)
            if X is None:
                continue
            X=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
            data_X=np.append(data_X,np.array(X),axis=0)
            data_y=np.append(data_y,np.array([Y]*len(X)),axis=0)
        data_X=data_X.reshape(data_X.shape[0],1,data_X.shape[1])
        data_y=data_y.reshape(data_y.shape[0],1,data_y.shape[1])
        return data_X,data_y



if __name__=="__main__":
   # gpus=tf.config.experimental.list_physical_devices("GPU")
   # tf.config.experimental.set_visible_devices(gpus[0],"GPU")
    position_to_num={
        "TOP":0,
        "JUNGLE":1,
        "MIDDLE":2,
        "BOTTOM":3,
        "UTILITY":4
    }
    data_path="./match_data_timeline_include_item_id"
    model_path="./backup/7500.h5"
    test_data_size=100 #모델 평가를 위한 데이터 개수
    model=load_model(model_path)
    gen=DataGen(data_path,100,test_data_size)
    loss_and_metric=model.evaluate(gen,verbose=1)
    print(str(loss_and_metric))