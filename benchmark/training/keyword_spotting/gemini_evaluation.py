import os
import numpy as np
import subprocess  # To run external commands
from sklearn.metrics import accuracy_score, roc_auc_score

import get_dataset as kws_data
import kws_util
import eval_functions_eembc as eembc_ev

num_classes = 12  # Modify as necessary

def run_gemini(gemini_path, model_path, sample_path, output_file="/work1/gitlab-runner-docker-data/ATHENAP18/FROM_ST_TO_ASYGN/tensorflow_gemini/geminipyc128PE/gmn_out/output_gemini.txt"):
    """
    Run the gemini executable with the given model and sample, and read the output.
    """
    # Construct the command with the full path to the gemini executable
    cmd = [gemini_path, model_path, sample_path]

    # Run the command
    subprocess.run(cmd)

    # Read predictions from the output file (output_gemini.txt)
    with open(output_file, 'r') as f:
        predictions = [int(line.strip()) for line in f.readlines()]

    return predictions

def save_sample_with_shape(sample, sample_path):

    shape = sample.shape
    directory = os.path.dirname(sample_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(sample_path, 'w') as f:
        f.write(f"{shape[1]} {shape[0]} {shape[2]}\n")

        np.savetxt(f, sample.reshape(-1, shape[0]), fmt="%i", delimiter=' ')

if __name__ == '__main__':
    Flags, unparsed = kws_util.parse_command()

    print('We will download data to {:}'.format(Flags.data_dir))

    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
    print("Done getting data")

    gemini_path = '/work1/gitlab-runner-docker-data/ATHENAP18/FROM_ST_TO_ASYGN/tensorflow_gemini/geminipyc128PE/gemini'

    model_path = '/work1/gitlab-runner-docker-data/ATHENAP18/FROM_ST_TO_ASYGN/tensorflow_gemini/geminipyc128PE/txt/model_unknown.txt'
    sample_path = '/work1/gitlab-runner-docker-data/ATHENAP18/FROM_ST_TO_ASYGN/tensorflow_gemini/geminipyc128PE/txt/kws.txt'
    
    all_predictions = []
    all_labels = []

    for i, (samples, batch_labels) in enumerate(ds_test):
        for j, sample in enumerate(samples):
            save_sample_with_shape(sample.numpy(), sample_path)

            gemini_predictions = run_gemini(gemini_path, model_path, sample_path)


            all_predictions.extend(gemini_predictions)
            all_labels.extend(batch_labels.numpy())

        print(f"Processed batch {i + 1}/{len(ds_test)}")


    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Accuracy: {accuracy}")

    
    auc_scikit = roc_auc_score(all_labels, np.eye(num_classes)[all_predictions], multi_class='ovr')
    print(f"AUC (sklearn) = {auc_scikit}")

    
    print("==== EEMBC calculate_accuracy Method ====")
    accuracy_eembc = eembc_ev.calculate_accuracy(np.eye(num_classes)[all_predictions], all_labels)
    print(40 * "=")

    print("==== EEMBC calculate_auc ====")
    label_names = ["go", "left", "no", "off", "on", "right",
                   "stop", "up", "yes", "silence", "unknown"]
    auc_eembc = eembc_ev.calculate_auc(np.eye(num_classes)[all_predictions], all_labels, label_names, Flags.model_architecture)
    print("---------------------")
