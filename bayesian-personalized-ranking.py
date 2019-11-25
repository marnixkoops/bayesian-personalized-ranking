####################################################################################################
# 🚀 IMPORTS
####################################################################################################

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp
import time

from tensorflow.python.client import device_lib
from tqdm import tqdm

####################################################################################################
# 🚀 EXPERIMENT SETTINGS
####################################################################################################

all_devices = str(device_lib.list_local_devices())
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if "GPU" in all_devices:
    DEVICE = "GPU"
    MACHINE = "Cloud VM"
elif "CPU" in all_devices:
    DEVICE = "CPU"
    MACHINE = "Local Machine"

print("🧠 Running TensorFlow version {} on {}".format(tf.__version__, DEVICE))

# model constants
epochs = 48
batches = 64
n_latent_varss = 64  # number of latent features in the MF
n_triplets = 5000  # how many (u,i,j) triplets we sample for each batch

# lambda regularization strength
lambda_cookie_id = 0.001
lambda_item = 0.001
lambda_bias = 0.001
learning_rate = 0.005


####################################################################################################
# 🚀 PREPARE DATA
####################################################################################################

t_prep = time.time()  # start timer for preparing data

# load input data
df = pd.read_csv("./data/product_sequences.csv")
df.head(3)

product_map_df = pd.read_csv("./data/product_mapping.csv")  # product id to name key-values
product_map_df["product_id"] = product_map_df["product_id"].astype(str)

# split data into train and test parition
train_partition_final_row = int(0.8 * len(df))
df_test = df[train_partition_final_row:]
df = df[:train_partition_final_row].copy()

df = pd.DataFrame(df["product_sequence"].str.split(",").tolist(), index=df["cookie_id"]).stack()
df = df.reset_index([0, "cookie_id"])
df.columns = ["cookie_id", "product_id"]
df = df.groupby(df.columns.tolist(), as_index=False).size()
df = df.reset_index(drop=False)
df.columns = ["cookie_id", "product_id", "clicks"]
df = df.dropna()
df.head(3)

# Convert product_ids names into integer ids
df["cookie_token"] = df["cookie_id"].astype("category").cat.codes
df["product_token"] = df["product_id"].astype("category").cat.codes

# Create a lookup frame so we can get the product_ids back later
item_lookup = df[["product_token", "product_id"]].drop_duplicates()
item_lookup["product_token"] = item_lookup.product_token.astype(str)
df = df.drop(["cookie_id", "product_id"], axis=1)
df = df.loc[df.clicks != 0]  # drop sessions without views (contain no information)
df.head(3)

# lists of all cookie_ids, product_ids and clicks
cookie_ids = list(np.sort(df.cookie_token.unique()))
product_ids = list(np.sort(df.product_token.unique()))
clicks = list(df.clicks)

# rows and columns for our new matrix
rows = df.cookie_token.astype(float)
cols = df.product_token.astype(float)

# contruct a sparse matrix for our cookie_ids and items containing number of clicks
data_sparse = sp.csr_matrix((clicks, (rows, cols)), shape=(len(cookie_ids), len(product_ids)))
uids, iids = data_sparse.nonzero()

print("⏱️ Elapsed time for processing input data: {:.3} seconds".format(time.time() - t_prep))


####################################################################################################
# 🚀 DEFINE TENSORFLOW GRAPH
####################################################################################################

t_train = time.time()  # start timer for training
graph = tf.Graph()


def init_variable(size, dim, name=None):
    """
    Helper function to initialize a new variable with
    uniform random values.
    """
    std = np.sqrt(2 / dim)
    return tf.Variable(tf.random_uniform([size, dim], -std, std), name=name)


def embed(inputs, size, dim, name=None):
    """
    Helper function to get a Tensorflow variable and create
    an embedding lookup to map our cookie_id and item
    indices to vectors.
    """
    emb = init_variable(size, dim, name)
    return tf.nn.embedding_lookup(emb, inputs)


def get_variable(graph, session, name):
    """
    Helper function to get the value of a
    Tensorflow variable by name.
    """
    v = graph.get_operation_by_name(name)
    v = v.values()[0]
    v = v.eval(session=session)
    return v


with graph.as_default():
    """
    Loss function:
    -SUM ln σ(xui - xuj) + λ(w1)**2 + λ(w2)**2 + λ(w3)**2 ...
    ln = the natural log
    σ(xuij) = the sigmoid function of xuij.
    λ = lambda regularization value.
    ||W||**2 = the squared L2 norm of our model parameters.

    """

    # Input into our model,  cookie_id (u), known item (i) an unknown item (i) triplets
    u = tf.placeholder(tf.int32, shape=(None, 1))
    i = tf.placeholder(tf.int32, shape=(None, 1))
    j = tf.placeholder(tf.int32, shape=(None, 1))

    # cookie_id feature embedding
    u_factors = embed(u, len(cookie_ids), n_latent_varss, "cookie_id_factors")  # U matrix

    # Known and unknown item embeddings
    item_factors = init_variable(len(product_ids), n_latent_varss, "item_factors")  # V matrix
    i_factors = tf.nn.embedding_lookup(item_factors, i)
    j_factors = tf.nn.embedding_lookup(item_factors, j)

    # i and j bias embeddings
    item_bias = init_variable(len(product_ids), 1, "item_bias")
    i_bias = tf.nn.embedding_lookup(item_bias, i)
    i_bias = tf.reshape(i_bias, [-1, 1])
    j_bias = tf.nn.embedding_lookup(item_bias, j)
    j_bias = tf.reshape(j_bias, [-1, 1])

    # Calculate the dot product + bias for known and unknown item to get xui and xuj
    xui = i_bias + tf.reduce_sum(u_factors * i_factors, axis=2)
    xuj = j_bias + tf.reduce_sum(u_factors * j_factors, axis=2)
    xuij = xui - xuj

    # Calculate the mean AUC (area under curve). If xuij is greater than 0, that means that xui is
    # greater than xuj (and thats what we want).
    u_auc = tf.reduce_mean(tf.to_float(xuij > 0))
    tf.summary.scalar("auc", u_auc)

    # Calculate the squared L2 norm ||W||**2 multiplied by λ
    l2_norm = tf.add_n(
        [
            lambda_cookie_id * tf.reduce_sum(tf.multiply(u_factors, u_factors)),
            lambda_item * tf.reduce_sum(tf.multiply(i_factors, i_factors)),
            lambda_item * tf.reduce_sum(tf.multiply(j_factors, j_factors)),
            lambda_bias * tf.reduce_sum(tf.multiply(i_bias, i_bias)),
            lambda_bias * tf.reduce_sum(tf.multiply(j_bias, j_bias)),
        ]
    )

    # Calculate the loss as ||W||**2 - ln σ(Xuij)
    # loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
    loss = -tf.reduce_mean(tf.log(tf.sigmoid(xuij))) + l2_norm

    # Train using the Adam optimizer to minimize our loss function
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    step = opt.minimize(loss)
    init = tf.global_variables_initializer()


####################################################################################################
# 🚀 MODEL TRAINING
####################################################################################################

# Run the session.
session = tf.Session(config=None, graph=graph)
session.run(init)

# This has noting to do with tensorflow but gives
# us a nice progress bar for the training.
progress = tqdm(total=batches * epochs)

for _ in range(epochs):
    for _ in range(batches):

        # We want to sample one known and one unknown
        # item for each cookie_id.

        # First we sample 15000 uniform indices.
        idx = np.random.randint(low=0, high=len(uids), size=n_triplets)

        # We then grab the cookie_ids matching those indices
        batch_u = uids[idx].reshape(-1, 1)

        # Then the known items for those cookie_ids
        batch_i = iids[idx].reshape(-1, 1)

        # Lastly we randomly sample one unknown item for each cookie_id
        batch_j = np.random.randint(
            low=0, high=len(product_ids), size=(n_triplets, 1), dtype="int32"
        )

        # Feed our cookie_ids, known and unknown items to
        # our tensorflow graph.
        feed_dict = {u: batch_u, i: batch_i, j: batch_j}

        # We run the session.
        _, l, auc = session.run([step, loss, u_auc], feed_dict)

    progress.update(batches)
    progress.set_description("Loss: %.3f | AUC: %.3f" % (l, auc))

progress.close()

train_time = time.time() - t_train
print(
    "⏱️ Elapsed time for training on {} sequences: {:.3} minutes".format(len(df), train_time / 60)
)

####################################################################################################
# 🚀 DEFINE SIMILARITY LOOK UP AND RECOMMENDATIONS
####################################################################################################


def find_similar_product_ids(product_id=None, num_items=15):
    """Find product_ids similar to an product_id.
    Args:
        product_id (str): The name of the product_id we want to find similar product_ids for
        num_items (int): How many similar product_ids we want to return.
    Returns:
        similar (pandas.DataFrame): DataFrame with num_items product_id names and scores
    """

    cookie_id_vecs = get_variable(graph, session, "cookie_id_factors")  # matrix U
    item_vecs = get_variable(graph, session, "item_factors")  # matrix V
    item_bias = get_variable(graph, session, "item_bias").reshape(-1)
    item_id = int(item_lookup[item_lookup.product_id == product_id]["product_token"])
    item_vec = item_vecs[item_id].T  # Transpose item vector

    # Calculate the similarity between this product and all other product_ids
    # by multiplying the item vector with our item_matrix
    scores = np.add(item_vecs.dot(item_vec), item_bias).reshape(1, -1)[0]
    top_10 = np.argsort(scores)[::-1][:num_items]  # Indices of top similarities

    # Map the indices to product_id names
    product_ids, product_id_scores = [], []

    for idx in top_10:
        product_ids.append(
            item_lookup.product_id.loc[item_lookup.product_token == str(idx)].iloc[0]
        )
        product_id_scores.append(scores[idx])

    similar = pd.DataFrame({"product_id": product_ids, "score": product_id_scores})
    similar["product_name"] = similar["product_id"].map(
        dict(zip(product_map_df["product_id"], product_map_df["product_name"]))
    )
    similar["product_type_name"] = similar["product_id"].map(
        dict(zip(product_map_df["product_id"], product_map_df["product_type_name"]))
    )

    return similar


def make_recommendation(cookie_token=None, num_items=5):
    """Recommend items for a given cookie_id given a trained model
    Args:
        cookie_token (int): The id of the cookie_id we want to create recommendations for.
        num_items (int): How many recommendations we want to return.
    Returns:
        recommendations (pandas.DataFrame): DataFrame with num_items product_id names and scores
    """

    # make df of the session for input token
    clicks = df[df["cookie_token"] == cookie_token].merge(
        item_lookup, on="product_token", how="left"
    )
    clicks["product_name"] = clicks["product_id"].map(
        dict(zip(product_map_df["product_id"], product_map_df["product_name"]))
    )
    clicks["product_type_name"] = clicks["product_id"].map(
        dict(zip(product_map_df["product_id"], product_map_df["product_type_name"]))
    )

    print("Making implicit feedback recommendations for observed user views: \n{}".format(clicks))
    print("\n----------------------⟶\n")

    cookie_id_vecs = get_variable(graph, session, "cookie_id_factors")  # matrix U
    item_vecs = get_variable(graph, session, "item_factors")  # matrix V
    item_bias = get_variable(graph, session, "item_bias").reshape(-1)
    rec_vector = np.add(cookie_id_vecs[cookie_token, :].dot(item_vecs.T), item_bias)
    item_idx = np.argsort(rec_vector)[::-1][:num_items]  # get indices of top cooki

    # Map the indices to product_id names
    product_ids, scores = [], []

    for idx in item_idx:
        product_ids.append(
            item_lookup.product_id.loc[item_lookup.product_token == str(idx)].iloc[0]
        )
        scores.append(rec_vector[idx])

    recommendations = pd.DataFrame({"product_id": product_ids, "score": scores})
    recommendations["product_name"] = recommendations["product_id"].map(
        dict(zip(product_map_df["product_id"], product_map_df["product_name"]))
    )
    recommendations["product_type_name"] = recommendations["product_id"].map(
        dict(zip(product_map_df["product_id"], product_map_df["product_type_name"]))
    )

    return recommendations


####################################################################################################
# 🚀 VALIDATE RESULTS
####################################################################################################

print(find_similar_product_ids(product_id="828805"))  # Airpods 2

print(find_similar_product_ids(product_id="838335"))  # iPhone 11

print(find_similar_product_ids(product_id="793672"))  # iPhone 8 Plus

print(find_similar_product_ids(product_id="817318"))  # Sony noise-cancelling

print(find_similar_product_ids(product_id="834996"))  # Macbook Pro 13" touch bar 2019

print(find_similar_product_ids(product_id="835003"))  # Macbook Air 13" 2019

print(find_similar_product_ids(product_id="795117"))  # doorbell

print(find_similar_product_ids(product_id="671720"))  # smart thermo

print(find_similar_product_ids(product_id="812182"))  # kobo e-reader

print(find_similar_product_ids(product_id="828471"))  # dyson vacuum


print(make_recommendation(cookie_token=3))

print(make_recommendation(cookie_token=4 ** 1))

print(make_recommendation(cookie_token=4 ** 5))
