from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from transformers import BatchFeature
import torch
from datasets import Dataset
import numpy as np
import pandas as pd
import os, pickle, glob
import argparse
import sys

from DF_transformer import DFFForImageClassification
from DF_transformer import DFFForPreTraining



def metrics_func(true_value, predict_value):
    # metrics
    r2 = metrics.r2_score(true_value, predict_value)
    mae = metrics.mean_absolute_error(true_value, predict_value)
    mse = metrics.mean_squared_error(true_value, predict_value)
    rmse = mse ** 0.5
    pearson_r = pearsonr(true_value, predict_value)[0]
    pearson_r2 = pearson_r ** 2
    rho, p_value = spearmanr(true_value, predict_value)
    tau, k_p_value = kendalltau(true_value, predict_value)

    # print
    # print(f"Metric - r2: {r2:.3f} mae: {mae:.3f} mse: {mse:.3f} "
    #       f"rmse: {rmse:.3f} pearsonr: {pearson_r:.3f} pearsonr2: {pearson_r2:.3f} spearman_rho: {rho:.3f} spearman_pvalue: {p_value:.3f} kendalltau: {tau:.3f} kendalltau_p: {k_p_value:.3f}")
    return r2, mae, mse, rmse, pearson_r, pearson_r2, tau


def get_predictions(pl_feature_array = None, scaler_path = r'./large.sav', model_path = r'./'):

    # data prepare and preprocess
    scaler = pickle.load(open(scaler_path, 'rb'))
    num_sample, num_channel, height, width = np.shape(pl_feature_array)
    data_0 = np.reshape(pl_feature_array, [num_sample, num_channel*height*width])
    scaled_data = scaler.transform(data_0).reshape([num_sample, num_channel, height, width])
    model_inputs = BatchFeature({"topological_features": scaled_data}, tensor_type='pt')

    # load model
    model = DFFForImageClassification.from_pretrained(model_path)

    # prediction
    with torch.no_grad():
        outputs = model(**model_inputs)
    predicted_value = outputs.logits.squeeze().numpy()

    return predicted_value


def main_get_predictions_for_scoring_CASF_2007():

    test_set=2007
    pl_feature_file = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_benchmars_features/all_feature_ele_scheme_1-dff0_domine20-6channel-cutoff010.npy'
    test_label_file = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_benchmars_features/downstream_task_labels/CASF2007_core_test_label.csv'
    scaler_path = r'/home/chendo11/workfolder/Pfizer_project/DFFormer_package/DFFormer/code_pkg/pretrain_data_standard_minmax_6channel_large.sav'
    model_pathes = glob.glob(r'/home/chendo11/workfolder/Pfizer_project/output_models/finetuned_pls_model_by_2020_64_0.0008/model_pls_cls_*')
    save_result_as_csv = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_results/casf2007_from_v2020finetuend_score.csv'

    # pl feature
    pl_feature_dict = np.load(pl_feature_file, allow_pickle=True).item()
    label_df = pd.read_csv(test_label_file, header=0, index_col=0)
    pl_feature_array = [pl_feature_dict[key] for key in label_df.index.tolist()]

    # prediction
    all_predictions = {}
    for i, model_p in enumerate(model_pathes):
        print(i, model_p)
        predicted_value = get_predictions(
            pl_feature_array=pl_feature_array, scaler_path=scaler_path, model_path=model_p)
        all_predictions[f'model_{i}'] = predicted_value
    
    df = pd.DataFrame(all_predictions)
    df.to_csv(save_result_as_csv)
    return None


def main_get_predictions_for_scoring_CASF_2013():

    test_set=2007
    pl_feature_file = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_benchmars_features/all_feature_ele_scheme_1-dff0_domine20-6channel-cutoff010.npy'
    test_label_file = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_benchmars_features/downstream_task_labels/CASF2013_core_test_label.csv'
    scaler_path = r'/home/chendo11/workfolder/Pfizer_project/DFFormer_package/DFFormer/code_pkg/pretrain_data_standard_minmax_6channel_large.sav'
    model_pathes = glob.glob(r'/home/chendo11/workfolder/Pfizer_project/output_models/finetuned_pls_model_by_2020_64_0.0008/model_pls_cls_*')
    save_result_as_csv = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_results/casf2013_from_v2020finetuend_score.csv'

    # pl feature
    pl_feature_dict = np.load(pl_feature_file, allow_pickle=True).item()
    label_df = pd.read_csv(test_label_file, header=0, index_col=0)
    pl_feature_array = [pl_feature_dict[key] for key in label_df.index.tolist()]

    # prediction
    all_predictions = {}
    for i, model_p in enumerate(model_pathes):
        print(i, model_p)
        predicted_value = get_predictions(
            pl_feature_array=pl_feature_array, scaler_path=scaler_path, model_path=model_p)
        all_predictions[f'model_{i}'] = predicted_value
    
    df = pd.DataFrame(all_predictions)
    df.to_csv(save_result_as_csv)
    return None


def main_get_predictions_for_scoring_CASF_2016():

    test_set=2007
    pl_feature_file = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_benchmars_features/all_feature_ele_scheme_1-dff0_domine20-6channel-cutoff010.npy'
    test_label_file = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_benchmars_features/downstream_task_labels/CASF2016_core_test_label.csv'
    scaler_path = r'/home/chendo11/workfolder/Pfizer_project/DFFormer_package/DFFormer/code_pkg/pretrain_data_standard_minmax_6channel_large.sav'
    model_pathes = glob.glob(r'/home/chendo11/workfolder/Pfizer_project/output_models/finetuned_pls_model_by_2020_64_0.0008/model_pls_cls_*')
    save_result_as_csv = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_results/casf2016_from_v2020finetuend_score.csv'

    # pl feature
    pl_feature_dict = np.load(pl_feature_file, allow_pickle=True).item()
    label_df = pd.read_csv(test_label_file, header=0, index_col=0)
    pl_feature_array = [pl_feature_dict[key] for key in label_df.index.tolist()]

    # prediction
    all_predictions = {}
    for i, model_p in enumerate(model_pathes):
        print(i, model_p)
        predicted_value = get_predictions(
            pl_feature_array=pl_feature_array, scaler_path=scaler_path, model_path=model_p)
        all_predictions[f'model_{i}'] = predicted_value
    
    df = pd.DataFrame(all_predictions)
    df.to_csv(save_result_as_csv)
    return None


def main_get_predictions_for_scoring_v_2016():

    test_set=2007
    pl_feature_file = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_benchmars_features/all_feature_ele_scheme_1-dff0_domine20-6channel-cutoff010.npy'
    test_label_file = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_benchmars_features/downstream_task_labels/v2016_core_test_label.csv'
    scaler_path = r'/home/chendo11/workfolder/Pfizer_project/DFFormer_package/DFFormer/code_pkg/pretrain_data_standard_minmax_6channel_large.sav'
    model_pathes = glob.glob(r'/home/chendo11/workfolder/Pfizer_project/output_models/finetuned_pls_model_by_2020_64_0.0008/model_pls_cls_*')
    save_result_as_csv = r'/home/chendo11/workfolder/Pfizer_project/pdbbind_results/v2016_from_v2020finetuend_score.csv'

    # pl feature
    pl_feature_dict = np.load(pl_feature_file, allow_pickle=True).item()
    label_df = pd.read_csv(test_label_file, header=0, index_col=0)
    pl_feature_array = [pl_feature_dict[key] for key in label_df.index.tolist()]

    # prediction
    all_predictions = {}
    for i, model_p in enumerate(model_pathes):
        print(i, model_p)
        predicted_value = get_predictions(
            pl_feature_array=pl_feature_array, scaler_path=scaler_path, model_path=model_p)
        all_predictions[f'model_{i}'] = predicted_value
    
    df = pd.DataFrame(all_predictions)
    df.to_csv(save_result_as_csv)
    return None


def main():
    # get_predictions()
    main_get_predictions_for_scoring_CASF_2007()
    main_get_predictions_for_scoring_CASF_2013()
    main_get_predictions_for_scoring_CASF_2016()
    main_get_predictions_for_scoring_v_2016()
    return None


if __name__ == "__main__":
    main()
    print('End!')