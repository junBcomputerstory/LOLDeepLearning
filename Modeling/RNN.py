import numpy as np
import json
from keras.layers import SimpleRNN, Dense
import keras
import tensorflow as tf
import os
import pickle


def create_data(path: str):
    try:
        with open(path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            a = [
                [

                    [0] * 430 for __ in range(10)
                ] for ___ in range(50)
            ]
            champion_data = data["metadata"]["champions"]
            team_100_win = None
            perk_list = dict()
            for i in range(1, 11):
                index = str(i)
                perk_list[index] = champion_data[index]["perk"]
            previous_items_list = [list() for _ in range(5)]
            item_order = [
                list() for _ in range(5)
            ]
            frame_cnt = 0
            for frame in data["info"]["frames"]:
                if (frame_cnt > 50):
                    break
                frame_data = a[frame_cnt]
                frame_info = frame["participantFrames"]
                for who in frame_info:
                    index = position_to_num[champion_data[who]["position"]]
                    if champion_data[who]["teamId"] == 200:
                        team_100_win = (champion_data[who]["win"] == False)
                        index += 5
                    champion_index = champion_data[who]["champion"]
                    item_list = frame_info[who]["items"]
                    frame_data[index][champion_index] = 1
                    for i in range(len(item_list)):
                        item_list[i] += 1

                    item_list.sort()
                    frame_data[index][428] = frame_info[who]["currentGold"]
                    frame_data[index][429] = frame_info[who]["totalGold"]
                    for perk in perk_list[who]:
                        frame_data[index][perk] = 1
                    for item_id in item_list:
                        frame_data[index][230 + item_id] = 1
                    if index < 5 and previous_items_list[index] != item_list:
                        for item in item_list:
                            if item not in previous_items_list[index]:
                                item_order[index].append(item)
                        previous_items_list[index] = item_list
                frame_cnt += 1
            data_x = np.array(a[:frame_cnt])
            data_y = [
                [
                    [0] * 230 for _ in range(25)
                ] for __ in range(5)
            ]
            for position in range(5):
                for order, item in enumerate(item_order[position]):
                    data_y[position][order][item] = 1
            data_y = np.array(data_y)
            if team_100_win:
                data_y = np.array([1, 0])
            else:
                data_y = np.array([0, 1])
            return data_x, data_y
    except:
        return None, None


if __name__ == "__main__":
    position_to_num = {
        "TOP": 0,
        "JUNGLE": 1,
        "MIDDLE": 2,
        "BOTTOM": 3,
        "UTILITY": 4
    }
    data = None
    data_path = "./match_data_timeline_include_item_id"
    data_set_list = os.listdir(data_path)
    train_size = int(len(data_set_list) * 0.7)
    train_data = data_set_list[:train_size]
    test_size = data_set_list[train_size:]
    X, y = create_data(data_path + "/" + train_data[0])
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = y.reshape(-1)
    print(y.shape)
    model = keras.Sequential()
    '''
    model.add(GRU(10,input_shape=(X.shape[1],1)))
    model.add(Dense(50,activation="relu"))
    model.add(Dense(30,activation="relu"))
    model.add(Dense(50,activation="relu"))
    model.add(Dense(20,activation="relu"))
    '''
    model.add(Dense(200, input_shape=(1, X.shape[1]), activation="tanh"))
    model.add(SimpleRNN(60, activation="tanh", return_sequences=True))
    model.add(SimpleRNN(80, activation="tanh", return_sequences=True))
    model.add(SimpleRNN(100, activation="tanh", return_sequences=True))
    model.add(SimpleRNN(50, activation="tanh", return_sequences=True))
    model.add(SimpleRNN(20, activation="tanh", return_sequences=True))
    model.add(Dense(y.shape[0], activation="softmax"))
    es = keras.callbacks.EarlyStopping(monitor="categorical_accuracy", mode=max, patience=50)
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    model.build()
    model.summary()
    i = 4000
    for start in range(4000, len(data_set_list),4500):
        data_X = np.empty((0, 4300))
        data_y = np.empty((0, 2))
        for train_index in range(start, min(len(data_set_list), start + 500)):
            train = data_set_list[train_index]

            X, Y = create_data(data_path + "/" + train)
            if X is None:
                continue
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
            data_X = np.append(data_X, np.array(X), axis=0)
            data_y = np.append(data_y, np.array([Y] * len(X)), axis=0)
        data_X = data_X.reshape(data_X.shape[0], 1, data_X.shape[1])
        data_y = data_y.reshape(data_y.shape[0], 1, data_y.shape[1])
        history = model.fit(data_X, data_y, epochs=360, validation_split=0.3)
        i = i + 500
        with open("./backup/histoty_" + str(i), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        model.save("./backup/" + str(i) + ".h5")
