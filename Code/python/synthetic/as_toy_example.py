# Steps:
# Classifier based on data points
# Run AS to select new points
# Improved classification accuracy
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV as gscv

from utils import math_utils


from matplotlib import pyplot as plt


import IPython


def generate_classification_data(
    npts, dim=15, n_informative=10, n_redundant=2, n_classes=2, least_prev=0.1,
    n_clusters_per_class=2, flip_y=0.05):#, scale=1.0, shift=0.):

  if least_prev:
    other_prev = 1.0 - least_prev
    weights = [other_prev / (n_classes - 1)] * (n_classes - 1) + [least_prev]
  else:
    weights = None
  
  n_features = max(dim, n_informative + n_redundant)
  X, Y = make_classification(
      npts, n_features=n_features, n_informative=n_informative,
      n_redundant=n_redundant, n_classes=n_classes, weights=weights,
      n_clusters_per_class=n_clusters_per_class, flip_y=flip_y)

  ft = np.random.randn(n_features)
  ft = ft / np.linalg.norm(ft)
  return X, Y, ft


def generate_mv_fts(n_views, dim):
  view_fts = np.random.randn(n_views, dim)
  view_fts = view_fts / np.linalg.norm(view_fts, axis=1)[:, None]

  return view_fts


def generate_data_for_view(X, ft, v_ft, view_dim=None, max_seen=0.8):
  npts, dim = X.shape
  max_seen_dim = int(max_seen * dim)

  if view_dim is None:
    view_dim = dim
  similarity = ((1 + ft.dot(v_ft)) / 2)**1.5
  # print(np.linalg.norm(ft))
  # print(np.linalg.norm(v_ft))
  # print(similarity)
  v_seen_dim = max(1, int(max_seen_dim * similarity))
  v_seen_dim = min(dim, view_dim, v_seen_dim)
  v_rest_dim = view_dim - v_seen_dim

  R = math_utils.random_unitary_matrix(dim)[:, :v_seen_dim]
  X_view = X.dot(R)
  # print(v_seen_dim)
  if v_rest_dim > 0:
    X_random = np.random.randn(npts, v_rest_dim)
    X_view = np.concatenate([X_view, X_random], axis=1)

  return X_view


def evaluate_model(classifier, tr_x, tr_y, te_x, te_y):
  classifier.fit(tr_x, tr_y)
  tr_pred = classifier.predict(tr_x)
  te_pred = classifier.predict(te_x)

  tr_acc = (tr_pred == tr_y).sum() / tr_y.shape[0]
  te_acc = (te_pred == te_y).sum() / te_y.shape[0]

  return (tr_acc, te_acc)


def as_toy_example():
  n_views = 100
  npts = 1000
  dim = 20
  n_informative = 20
  n_redundant = 0
  n_classes = 3
  least_prev = None
  n_clusters_per_class = 3
  flip_y = 0.00

  X, Y, ft = generate_classification_data(
      npts=npts, dim=dim, n_informative=n_informative, n_redundant=n_redundant,
      n_classes=n_classes, least_prev=least_prev,
      n_clusters_per_class=n_clusters_per_class, flip_y=flip_y)

  dim = X.shape[1]
  view_dim = 10
  v_fts = generate_mv_fts(n_views, dim)

  tr_frac = 0.8
  ntr = int(npts * tr_frac)
  tr_y = y_tr = Y[:ntr]
  te_y = y_te = Y[ntr:]

  n_iters = 30
  max_seen = 0.7
  classifier = RandomForestClassifier(n_estimators=50)

  start_view_sim_position_frac = 0.85
  start_view_sim_position = int(start_view_sim_position_frac * (n_views - 1))

  similarity_to_base = v_fts.dot(ft)
  start_view = np.argsort(similarity_to_base)[start_view_sim_position]
  start_ft = v_fts[start_view]
  similarity_to_start = v_fts.dot(start_ft)

  X_start = generate_data_for_view(X, ft, start_ft, view_dim, max_seen)
  x_start_tr, x_start_te = X_start[:ntr], X_start[ntr:]
  tr_acc_start, te_acc_start = evaluate_model(
      classifier, x_start_tr, tr_y, x_start_te, te_y)

  view_Xs = {}

  # IPython.embed()
  # similarity order
  print("Picking in best similarity order...")
  print("Start idx: %i" % start_view)
  print("Base train acc of %.3f and test acc of %.3f"%
        (tr_acc_start, te_acc_start))

  Xs = X_start
  # sim_order = np.argsort(-similarity_to_start)
  sim_order = np.argsort(-similarity_to_base)[1:]
  sim_tr_accs = [tr_acc_start]
  sim_te_accs = [te_acc_start]
  sim_views_picked = [start_view]
  for idx in range(n_iters):
    view_id = sim_order[idx]
    view_ft = v_fts[view_id]
    if view_id not in view_Xs:
      view_Xs[view_id] = generate_data_for_view(X, ft, view_ft, view_dim, max_seen)
    X_view = view_Xs[view_id]

    Xs = np.concatenate([Xs, X_view], axis=1)
    vx_tr, vx_te = Xs[:ntr], Xs[ntr:]

    print("  Iteration %i: Picked view %i" % (idx + 1, view_id))
    print("  Evaluating model...", end="\r")

    v_tr_acc, v_te_acc = evaluate_model(
        classifier, vx_tr, tr_y, vx_te, te_y)
    print("  Current train acc of %.3f and test acc of %.3f\n"%
          (v_tr_acc, v_te_acc))
    sim_tr_accs.append(v_tr_acc)
    sim_te_accs.append(v_te_acc)

  ##############################################################################
  # AS Order
  # similarity order
  # IPython.embed()
  print("Picking in similarity (to start) order...")
  print("Start idx: %i" % start_view)
  print("Base train acc of %.3f and test acc of %.3f"%
        (tr_acc_start, te_acc_start))

  Xs = X_start
  # sim_order = np.argsort(-similarity_to_start)
  as_order = np.argsort(-similarity_to_start)[1:]
  as_tr_accs = [tr_acc_start]
  as_te_accs = [te_acc_start]
  as_views_picked = [start_view]
  for idx in range(n_iters):
    view_id = as_order[idx]
    view_ft = v_fts[view_id]
    if view_id not in view_Xs:
      view_Xs[view_id] = generate_data_for_view(X, ft, view_ft, view_dim, max_seen)
    X_view = view_Xs[view_id]

    Xs = np.concatenate([Xs, X_view], axis=1)
    vx_tr, vx_te = Xs[:ntr], Xs[ntr:]

    print("  Iteration %i: Picked view %i" % (idx + 1, view_id))
    print("  Evaluating model...", end="\r")

    v_tr_acc, v_te_acc = evaluate_model(
        classifier, vx_tr, tr_y, vx_te, te_y)
    print("  Current train acc of %.3f and test acc of %.3f\n"%
          (v_tr_acc, v_te_acc))
    as_tr_accs.append(v_tr_acc)
    as_te_accs.append(v_te_acc)

  # IPython.embed()
  print("Picking in random order...")
  print("Start idx: %i" % start_view)
  print("Base train acc of %.3f and test acc of %.3f"%
        (tr_acc_start, te_acc_start))

  Xs = X_start
  random_order = (
      np.arange(start_view).tolist() +
      np.arange(start_view + 1, n_views).tolist())
  np.random.shuffle(random_order)
  # random_order = np.argsort(similarity_to_base)[:-1]
  random_tr_accs = [tr_acc_start]
  random_te_accs = [te_acc_start]
  random_views_picked = [start_view]
  for idx in range(n_iters):
    view_id = random_order[idx]
    view_ft = v_fts[view_id]
    # X_view = generate_data_for_view(X, ft, view_ft, view_dim, max_seen)
    if view_id not in view_Xs:
      view_Xs[view_id] = generate_data_for_view(X, ft, view_ft, view_dim, max_seen)
    X_view = view_Xs[view_id]
    Xs = np.concatenate([Xs, X_view], axis=1)
    vx_tr, vx_te = Xs[:ntr], Xs[ntr:]

    print("  Iteration %i: Picked view %i" % (idx + 1, view_id))
    print("  Evaluating model...", end="\r")

    v_tr_acc, v_te_acc = evaluate_model(
        classifier, vx_tr, tr_y, vx_te, te_y)
    print("  Current train acc of %.3f and test acc of %.3f\n"%
          (v_tr_acc, v_te_acc))
    random_tr_accs.append(v_tr_acc)
    random_te_accs.append(v_te_acc)

  # IPython.embed()

  # p1 = sim_te_accs
  # p2 = random_te_accs
  # plt.plot(p1, label="AS")
  # plt.plot(p2, label="Random")

  # IPython.embed()
  return sim_te_accs, as_te_accs, random_te_accs


def as_toy_example_ntimes(n=25):
  all_sim = []
  all_as = []
  all_random = []

  for expt in range(n):
    sim_te_accs, as_te_accs, random_te_accs = as_toy_example()
    all_sim.append(np.array(sim_te_accs))
    all_as.append(np.array(as_te_accs))
    all_random.append(np.array(random_te_accs))

  all_sim = np.array(all_sim)
  all_as = np.array(all_as)
  all_random = np.array(all_random)

  avg_sim = all_sim.mean(0)
  avg_as = all_as.mean(0)
  avg_rand = all_random.mean(0)

  IPython.embed()

  return all_sim, all_as, all_random

if __name__ == "__main__":
  n = 25
  results = as_toy_example_ntimes(n)

  IPython.embed()
  nplot = 31

  nplot = min(nplot, avg_sim.shape[0])
  vrange = np.arange(nplot).astype(int)
  plt.plot(avg_sim[:nplot], label="Best order")
  plt.plot(avg_as[:nplot], label="AS order")
  plt.plot(avg_rand[:nplot], label="Random order")
  plt.ylabel("Classification accuracy", fontsize=30)
  plt.xlabel("Number of views", fontsize=30)
  plt.xticks(fontsize=15)
  if nplot < 10:
    plt.xticks(vrange, vrange)
  plt.yticks(fontsize=15)
  plt.title("View Selection: Classification Accuracy [25 run avg]", fontsize=40)
  plt.legend(fontsize=20)
  plt.show()
