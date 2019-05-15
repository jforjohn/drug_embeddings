from input_output.load_config import load_config_file
from MainLauncher import embedding_step, run_model_eval
import argparse
import sys
from os import mkdir, path
from math import log, sqrt
import pandas as pd
from collections import OrderedDict

def emb_experiment(config):
  emb_types = ['simple', 'w2v_sg', 'w2v_cb', 'pretrained_glove_50', 'pretrained_glove_200', 'pretrained_pub'] 
  emb_pretrained = ['', '', '', 'glove.6B.50d.txt', 'glove.6B.200d.txt', 'wikipedia-pubmed-and-PMC-w2v.bin']
  emb_alpha = [[0.001], [0.5], [0.5], [0], [0], [0]]
  emb_windows = [[0], [2, 5, 10], [2, 5, 10], [0], [0], [0]] 
  emb_dim = [[20, 50, 200], [20, 50, 200], [20, 50, 200], [50],[200], [200]]
  emb_epochs = [70, 200, 200, 0, 0, 0]

  config_emb = config['embeddings']
  output_dir_init = config['data']['output_dir']
  try:
    # Create target Directory
    mkdir(output_dir_init)
  except FileExistsError:
    print("Directory " , output_dir_init ,  " already exists")

  for ind in range(len(emb_types)):
    config_emb['emb_type'] = emb_types[ind]
    config_emb['emb_file'] = emb_pretrained[ind]
    #config_emb['alpha'] = emb_alpha[ind]
    config_emb['epochs'] = emb_epochs[ind]
    alphas = emb_alpha[ind]
    windows = emb_windows[ind]
    dims = emb_dim[ind]

    for ind_w in range(len(windows)):
      config_emb['window'] = windows[ind_w]
      config_emb['alpha'] = alphas[0]
      for dim in dims:
        config_emb['emb_dim'] = dim
        
        config['embeddings'] = config_emb
        (X_train, y_train, X_test, y_test, df_train, df_test, word2idx, tag2idx, emb_duration, weights_file, weights, n_words, n_tags, max_len) = embedding_step(config)

        weights_result_file = weights_file[:-4]
        results_dir = path.join(output_dir_init, weights_result_file)
        config['data']['output_dir'] = results_dir
        try:
          # Create target Directory
          mkdir(results_dir)
        except FileExistsError:
          print("Directory " , results_dir ,  " already exists")
        
        (f1s_model, score_trn, score_tst) = run_model_eval(
                      X_train, y_train, X_test, y_test,
                      df_test, config, tag2idx, n_words,
                      n_tags, max_len, weights=weights)

        # write results
        dict_results = OrderedDict({
          'Out File': weights_result_file,
          'Type': config_emb['emb_type'],
          'File': config_emb['emb_file'],
          'LR': config_emb['alpha'],
          'Window': config_emb['window'],
          'Dim': config_emb['emb_dim'],
          'Max len': max_len,
          'Tr loss': score_trn[0],
          'Tst loss': score_tst[0],
          'Tr acc': score_trn[1],
          'Tst acc': score_tst[0],
          'F1s': f1s_model
        })
        df_results = pd.DataFrame([dict_results])
        df_results.to_csv('results_exp.csv', mode='a', header=False, index=False)
        

if __name__ == '__main__':
  # Loads config
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-c", "--config", default="config.json",
      help="specify the location of the clustering config file"
  )
  args, _ = parser.parse_known_args()

  config_file = args.config
  config = load_config_file(config_file, args) # args

  emb_experiment(config)
  #(X_train, y_train, X_test, y_test, df_train, df_test, word2idx, tag2idx, emb_duration, weights, n_words, n_tags, max_len) = embedding_step(config)

