import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import os


class Embedding:
    model = ...  # type: Model

    def __init__(self, model: tf.keras.models.Model):
        self.model = model

    def get_embedding(self, items):
        embedding = []
        paths = []

        for item in items:
            try:
                prep = self.prep_item(item)
                #prep = self.prep_item(item[0])
            except:
                continue
            batch = np.array([prep])
            features = self.model.predict_on_batch(batch)

            # embedding, productid
            tuple = (features, item[:-4])

            # embedding, categoryid
            #tuple = (features, item[1])
            embedding.append(tuple)
            paths.append(item)

        return embedding, paths

    def get_features(self, path_to_img):

        preprocessed = self.prep_item(path_to_img)
        features = self.model.predict(preprocessed)
        return features

    def collect_data(self, dataset_path, items_count_limit=-1):
        """
        Load data from json
        """
        with open(dataset_path) as json_file:
            data = json.load(json_file)
            result = []
            for outfit in data:
                for item in outfit['items']:
                    if items_count_limit != -1 and len(result) >= items_count_limit:
                        print("Dataset has been successfully loaded (" + str(len(result)) + " items).")
                        return result
                    tuple = (str(item["set_id"]) + "/" + str(item["index"]) + ".jpg", item['categoryid'])
                    result.append(tuple)

        print("Dataset has been successfully loaded (" + str(len(result)) + " items).")
        return result

    def collect_amazon_images(self, dirName):
        listOfFile = os.listdir(dirName)
        return listOfFile

    @staticmethod
    def prep_item(path_to_img: str):
        """
        Transforms dataset item into the model input
        :param path_to_img:
        :param item: Dataset item
        :return: Model input
        """
        img = tf.keras.preprocessing.image.load_img(
            Path("./data/colours/" + path_to_img), target_size=(299, 299))
        x = tf.keras.preprocessing.image.img_to_array(img)
        return tf.keras.applications.inception_v3.preprocess_input(x)

# target = model.model_target  # type: tf.keras.models.Model
# target.load_weights('../../logs/20200621-173230/model.h5', True)
# embedding = Embedding(target)
# data = embedding.collect_data("../../data/label/valid_no_dup_new.json")
# emb, paths = embedding.get_embedding(data)

# emb_array = np.array(emb)
# paths_array = np.array(paths)
# np.save("../../logs/20200621-173230/embeding", emb_array)
# np.save("../../logs/20200621-173230/paths", paths_array)
