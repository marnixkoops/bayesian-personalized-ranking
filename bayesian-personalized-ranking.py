import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp
import time

from tqdm import tqdm

tf.__version__  # needs TF 1.15


# ---------------------------
# LOAD AND PREPARE THE DATA
# ---------------------------

t_prep = time.time()  # start timer for preparing data

# input data
DATA_PATH1 = (
    "/Users/marnix.koops/Projects/marnix-single-flow-rnn/data/ga_product_sequence_20191013.csv"
)
DATA_PATH2 = (
    "/Users/marnix.koops/Projects/marnix-single-flow-rnn/data/ga_product_sequence_20191020.csv"
)
DATA_PATH3 = (
    "/Users/marnix.koops/Projects/marnix-single-flow-rnn/data/ga_product_sequence_20191027.csv"
)
DATA_PATH4 = (
    "/Users/marnix.koops/Projects/marnix-single-flow-rnn/data/ga_product_sequence_20191103.csv"
)


sequence_df = pd.read_csv(DATA_PATH1)
sequence_df2 = pd.read_csv(DATA_PATH2)
sequence_df3 = pd.read_csv(DATA_PATH3)
sequence_df4 = pd.read_csv(DATA_PATH4)
df = sequence_df2.append(sequence_df2).append(sequence_df3).append(sequence_df4)
del sequence_df, sequence_df2, sequence_df3, sequence_df4
df = df.drop_duplicates(keep="first")  # also checks for visit_date + id
product_map_df = pd.read_csv(
    "/Users/marnix.koops/Projects/marnix-single-flow-rnn/data/product_mapping.csv"
)
product_map_df["product_id"] = product_map_df["product_id"].astype(str)

product_map_df.head()

train_partition_final_row = int(0.8 * len(df))
df_test = df[train_partition_final_row:]
df = df[:train_partition_final_row].copy()

df = pd.DataFrame(
    df["product_sequence"].str.split(",").tolist(), index=df["coolblue_cookie_id"]
).stack()
df = df.reset_index([0, "coolblue_cookie_id"])
df.columns = ["coolblue_cookie_id", "product_id"]
df = df.groupby(df.columns.tolist(), as_index=False).size()
df = df.reset_index(drop=False)
df.columns = ["coolblue_cookie_id", "product_id", "clicks"]
df.head()


# Drop any rows with missing values
df = df.dropna()

# Convert product_ids names into numerical IDs
df["coolblue_cookie_token"] = df["coolblue_cookie_id"].astype("category").cat.codes
df["product_token"] = df["product_id"].astype("category").cat.codes

# Create a lookup frame so we can get the product_id
# names back in readable form later.
item_lookup = df[["product_token", "product_id"]].drop_duplicates()
item_lookup["product_token"] = item_lookup.product_token.astype(str)

# We drop our old coolblue_cookie_id and product_id columns
df = df.drop(["coolblue_cookie_id", "product_id"], axis=1)

# Drop any rows with 0 clicks
df = df.loc[df.clicks != 0]

# Create lists of all coolblue_cookie_ids, product_ids and clicks
coolblue_cookie_ids = list(np.sort(df.coolblue_cookie_token.unique()))
product_ids = list(np.sort(df.product_token.unique()))
clicks = list(df.clicks)

# Get the rows and columns for our new matrix
rows = df.coolblue_cookie_token.astype(float)
cols = df.product_token.astype(float)

# Contruct a sparse matrix for our coolblue_cookie_ids and items containing number of clicks
data_sparse = sp.csr_matrix(
    (clicks, (rows, cols)), shape=(len(coolblue_cookie_ids), len(product_ids))
)


# Get the values of our matrix as a list of coolblue_cookie_id ids
# and item ids. Note that our litsts have the same length
# as each coolblue_cookie_id id repeats one time for each played product_id.
uids, iids = data_sparse.nonzero()

print("⏱️ Elapsed time for processing input data: {:.3} seconds".format(time.time() - t_prep))

# -------------
# HYPERPARAMS
# -------------

epochs = 48
batches = 64
num_factors = 64  # Number of latent features

# Independent lambda regularization values
# for coolblue_cookie_id, items and bias.
lambda_coolblue_cookie_id = 0.0000001
lambda_item = 0.0000001
lambda_bias = 0.0000001

# Our learning rate
lr = 0.005

# How many (u,i,j) triplets we sample for each batch
samples = 5000

# -------------------------
# TENSORFLOW GRAPH
# -------------------------

t_train = time.time()  # start timer for training


# Set up our Tensorflow graph
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
    an embedding lookup to map our coolblue_cookie_id and item
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

    # Input into our model, in this case our coolblue_cookie_id (u),
    # known item (i) an unknown item (i) triplets.
    u = tf.placeholder(tf.int32, shape=(None, 1))
    i = tf.placeholder(tf.int32, shape=(None, 1))
    j = tf.placeholder(tf.int32, shape=(None, 1))

    # coolblue_cookie_id feature embedding
    u_factors = embed(
        u, len(coolblue_cookie_ids), num_factors, "coolblue_cookie_id_factors"
    )  # U matrix

    # Known and unknown item embeddings
    item_factors = init_variable(len(product_ids), num_factors, "item_factors")  # V matrix
    i_factors = tf.nn.embedding_lookup(item_factors, i)
    j_factors = tf.nn.embedding_lookup(item_factors, j)

    # i and j bias embeddings.
    item_bias = init_variable(len(product_ids), 1, "item_bias")
    i_bias = tf.nn.embedding_lookup(item_bias, i)
    i_bias = tf.reshape(i_bias, [-1, 1])
    j_bias = tf.nn.embedding_lookup(item_bias, j)
    j_bias = tf.reshape(j_bias, [-1, 1])

    # Calculate the dot product + bias for known and unknown
    # item to get xui and xuj.
    xui = i_bias + tf.reduce_sum(u_factors * i_factors, axis=2)
    xuj = j_bias + tf.reduce_sum(u_factors * j_factors, axis=2)

    # We calculate xuij.
    xuij = xui - xuj

    # Calculate the mean AUC (area under curve).
    # if xuij is greater than 0, that means that
    # xui is greater than xuj (and thats what we want).
    u_auc = tf.reduce_mean(tf.to_float(xuij > 0))

    # Output the AUC value to tensorboard for monitoring.
    tf.summary.scalar("auc", u_auc)

    # Calculate the squared L2 norm ||W||**2 multiplied by λ.
    l2_norm = tf.add_n(
        [
            lambda_coolblue_cookie_id * tf.reduce_sum(tf.multiply(u_factors, u_factors)),
            lambda_item * tf.reduce_sum(tf.multiply(i_factors, i_factors)),
            lambda_item * tf.reduce_sum(tf.multiply(j_factors, j_factors)),
            lambda_bias * tf.reduce_sum(tf.multiply(i_bias, i_bias)),
            lambda_bias * tf.reduce_sum(tf.multiply(j_bias, j_bias)),
        ]
    )

    # Calculate the loss as ||W||**2 - ln σ(Xuij)
    # loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
    loss = -tf.reduce_mean(tf.log(tf.sigmoid(xuij))) + l2_norm

    # Train using the Adam optimizer to minimize
    # our loss function.
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    step = opt.minimize(loss)

    # Initialize all tensorflow variables.
    init = tf.global_variables_initializer()

# ------------------
# GRAPH EXECUTION
# ------------------

# Run the session.
session = tf.Session(config=None, graph=graph)
session.run(init)

# This has noting to do with tensorflow but gives
# us a nice progress bar for the training.
progress = tqdm(total=batches * epochs)

for _ in range(epochs):
    for _ in range(batches):

        # We want to sample one known and one unknown
        # item for each coolblue_cookie_id.

        # First we sample 15000 uniform indices.
        idx = np.random.randint(low=0, high=len(uids), size=samples)

        # We then grab the coolblue_cookie_ids matching those indices.
        batch_u = uids[idx].reshape(-1, 1)

        # Then the known items for those coolblue_cookie_ids.
        batch_i = iids[idx].reshape(-1, 1)

        # Lastly we randomly sample one unknown item for each coolblue_cookie_id.
        batch_j = np.random.randint(low=0, high=len(product_ids), size=(samples, 1), dtype="int32")

        # Feed our coolblue_cookie_ids, known and unknown items to
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

# -----------------------
# FIND SIMILAR product_idS
# -----------------------


def find_similar_product_ids(product_id=None, num_items=10):
    """Find product_ids similar to an product_id.
    Args:
        product_id (str): The name of the product_id we want to find similar product_ids for
        num_items (int): How many similar product_ids we want to return.
    Returns:
        similar (pandas.DataFrame): DataFrame with num_items product_id names and scores
    """

    # Grab our coolblue_cookie_id matrix U
    coolblue_cookie_id_vecs = get_variable(graph, session, "coolblue_cookie_id_factors")

    # Grab our Item matrix V
    item_vecs = get_variable(graph, session, "item_factors")

    # Grab our item bias
    item_bi = get_variable(graph, session, "item_bias").reshape(-1)

    # Get the item id for Lady GaGa
    item_id = int(item_lookup[item_lookup.product_id == product_id]["product_token"])

    # Get the item vector for our item_id and transpose it.
    item_vec = item_vecs[item_id].T

    # Calculate the similarity between Lady GaGa and all other product_ids
    # by multiplying the item vector with our item_matrix
    scores = np.add(item_vecs.dot(item_vec), item_bi).reshape(1, -1)[0]

    # Get the indices for the top 10 scores
    top_10 = np.argsort(scores)[::-1][:num_items]

    # We then use our lookup table to grab the names of these indices
    # and add it along with its score to a pandas dataframe.
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


# ---------------------
# MAKE RECOMMENDATION
# ---------------------


def make_recommendation(coolblue_cookie_token=None, num_items=10):
    """Recommend items for a given coolblue_cookie_id given a trained model
    Args:
        coolblue_cookie_token (int): The id of the coolblue_cookie_id we want to create recommendations for.
        num_items (int): How many recommendations we want to return.
    Returns:
        recommendations (pandas.DataFrame): DataFrame with num_items product_id names and scores
    """

    # Grab our coolblue_cookie_id matrix U
    coolblue_cookie_id_vecs = get_variable(graph, session, "coolblue_cookie_id_factors")

    # Grab our item matrix V
    item_vecs = get_variable(graph, session, "item_factors")

    # Grab our item bias
    item_bi = get_variable(graph, session, "item_bias").reshape(-1)

    # Calculate the score for our coolblue_cookie_id for all items.
    rec_vector = np.add(coolblue_cookie_id_vecs[coolblue_cookie_token, :].dot(item_vecs.T), item_bi)

    # Grab the indices of the top coolblue_cookie_ids
    item_idx = np.argsort(rec_vector)[::-1][:num_items]

    # Map the indices to product_id names and add to dataframe along with scores.
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


print(find_similar_product_ids(product_id="795117"))


print(make_recommendation(coolblue_cookie_token=56))
