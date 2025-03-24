import json
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import numpy as np

st.title("🎈 My new app")
data_choices = ['1_phase_to_ground', '2_phase_to_ground', '3_phase_to_ground']

data_option = st.selectbox(
    'Choose an data option:',  # Prompt
    data_choices  # List of choices
)

if data_option == '1_phase_to_ground': 
    data_details_options_to_choose_from = ['PCA/PCA_6_noGaussian','PCA/PCA_64_noGaussian',  'PCAwithGaussian05/PCA_6_Gaussian05']
elif data_option == '2_phase_to_ground': 
    data_details_options_to_choose_from = ['PCA/PCA_6_noGaussian']
elif data_option == '3_phase_to_ground':
    data_details_options_to_choose_from = ['PCA/PCA_6_noGaussian']

data_details_option = st.selectbox(
    'Choose an option:',  # Prompt
    data_details_options_to_choose_from  # List of choices
)

if data_option == '1_phase_to_ground' and data_details_option == 'PCA/PCA_6_noGaussian': 
    models_to_choose_from = ["QML_strongly_100layers_100epochs_0.001lr_32batch_adamoptimizer",
                             "QML_simplified_50layers_100epochs_0.001lr_32batch_adamoptimizer", 
                             "CML_LSTM_stan_0.01lr_50epochs_256batch_size_1layers_64hidden",
                             "CML_ConvNet_stan_0.001lr_50epochs_128batch_size", 
                             "CML_ResNet18_stan_0.001lr_50epochs_256batch_size", 
                             "CML_ResNet34_stan_0.001lr_50epochs_128batch_size"]
    
elif data_option == '2_phase_to_ground' and data_details_option == 'PCA/PCA_6_noGaussian': 
    models_to_choose_from = ['QML_strongly_100layers_100epochs_0.001lr_64batch_adamoptimizer',
                            'QML_simplified_50layers_100epochs_0.01lr_32batch_adamoptimizer',
                            'CML_LSTM_0.001lr_50epochs_128batch_size_2layers_128hidden',
                            'CML_ConvNet_0.01lr_50epochs_256batch_size', 
                            'CML_ResNet18_0.01lr_50epochs_128batch_size',
                            'CML_ResNet34_0.001lr_50epochs_128batch_size']

elif data_option == '3_phase_to_ground' and data_details_option == 'PCA/PCA_6_noGaussian':
    models_to_choose_from = ['QML_simplified_50layers_100epochs_0.01lr_64batch_adamoptimizer',
                            'QML_strongly_100layers_100epochs_0.001lr_64batch_adamoptimizer',
                            'CML_LSTM_0.001lr_50epochs_128batch_size_1layers_128hidden',
                            'CML_ConvNet_0.001lr_50epochs_256batch_size', 
                            'CML_ResNet18_0.001lr_50epochs_128batch_size'
                            'CML_ResNet34_0.001lr_50epochs_128batch_size']

elif data_option == '1_phase_to_ground' and data_details_option == 'PCAwithGaussian05/PCA_6_Gaussian05': 
    models_to_choose_from = ["QML_simplified_50layers_100epochs_0.01lr_64batch_adamoptimizer",
                             "QML_strongly_100layers_100epochs_0.001lr_64batch_adamoptimizer", 
                             "CML_LSTM_0.001lr_50epochs_128batch_size_1layers_128hidden",
                             "CML_ConvNet_0.001lr_50epochs_256batch_size", 
                             "CML_ResNet18_0.001lr_50epochs_128batch_size", 
                             "CML_ResNet34_0.001lr_50epochs_128batch_size"]
else: 
    models_to_choose_from = None

model_options = st.multiselect(
    "Select models:",
    models_to_choose_from
)

adv_from = st.selectbox(
    'Choose an model to get adv samples from:',  # Prompt
    models_to_choose_from  # List of choices
)

path_to_data = f'{data_option}/{data_details_option}'
list_of_models = [[path_to_data, model] for model in model_options]


adv_from = [path_to_data, adv_from]

attacks_to_view = ['fgsm', 'fgsm_targeted', 'pgd', 'pgd_targeted']

attack_type = st.selectbox(
    'Choose an model to get adv samples from:',  # Prompt
    attacks_to_view  # List of choices
)

dataset_type = data_option

# Define the metrics to plot (you can modify this list as needed)
metrics_to_plot = ["accuracy", "recall", 'precision', 'f1']  # Add other metrics as necessary

# st.write(list_of_models)
# Loop through the metrics to create a separate plot for each
for metric in metrics_to_plot:
    all_data = []
    # Loop through models and plot their performance for the current metric
    for model_selected in list_of_models:
        _, model_name = model_selected
        result_to_open = f'{adv_from[0]}/{adv_from[1]}/performance_of_{model_name}_on_{attack_type}_adv_from_{adv_from[1]}.json'
        
        # Print for debugging (optional)
        print(f"Opening: {result_to_open}")

        # Open the JSON file and extract metrics
        with open(result_to_open, "r") as f:
            metrics = json.load(f)

        eps_values = list(map(float, metrics['metrics'].keys()))  # Sorted epsilon values
        metric_values = [metrics['metrics'][str(eps)].get(metric, None) for eps in eps_values]

        # Collect data for plotting
        if None not in metric_values:
            for eps, value in zip(eps_values, metric_values):
                all_data.append({"Epsilon": eps, "Metric Value": value, "Model": model_name, "Metric": metric})

    # Convert collected data into a DataFrame for Plotly
    df = pd.DataFrame(all_data)

    # Create the Plotly figure for the current metric
    fig = px.line(
        df, x="Epsilon", y="Metric Value", color="Model", markers=True,
        title=f"{metric.title()} Performance",
        labels={"Epsilon": "Epsilon (ε)", "Metric Value": f"{metric.title()}"},
        hover_data={"Epsilon": True, "Metric Value": True, "Model": True, "Metric": True}
    )

    # Customize legend position
    fig.update_layout(
        legend=dict(
            orientation="h",  # Horizontal legend
            y=-0.2,  # Move legend down
            x=0.5,
            xanchor="center"
        ),
        yaxis=dict(range=[0, 1.1])  # Keep y-axis in range
    )
    st.plotly_chart(fig, use_container_width=True)

st.write('Plot CM')


chosen_models_to_plot_cm = st.multiselect(
    'Select model to plot for',
    models_to_choose_from
)

# chosen_attack_to_plot_cm = st.selectbox(
#     'Choose attack to plot', 
#     attacks_to_view
# )
    

for model_plot_cm in chosen_models_to_plot_cm: 
    # st.write(metrics['metrics']['0.0'])
    with st.expander(model_plot_cm):
        result_to_open = f'{adv_from[0]}/{adv_from[1]}/performance_of_{chosen_models_to_plot_cm[0]}_on_{attack_type}_adv_from_{adv_from[1]}.json'

        # Open the JSON file and extract metrics
        with open(result_to_open, "r") as f:
            metrics = json.load(f)

        eps_values = list(map(float, metrics['metrics'].keys()))  # Sorted epsilon values
        cm_to_plot = [metrics['metrics'][str(eps)]['cm'] for eps in eps_values]

        for eps_idx, cm in enumerate(cm_to_plot): 
            # Plot the confusion matrix using Seaborn and Matplotlib
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', ax=ax)

            # Labels and title
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title(f'Confusion Matrix - {chosen_models_to_plot_cm[0]} - {eps_values[eps_idx]}')

            # Display the plot in the Streamlit app
            st.pyplot(fig)

def plot_adversarial_examples(X_test, X_test_adv, y_test=None, num_samples=2, eps=None, plot_labels=None):
    """
    Plots the original and adversarial examples for selected samples, with each feature shown as a subplot.

    Parameters:
        X_test (array): Original test samples (shape: samples, time steps, features).
        X_test_adv (array): Adversarial test samples (same shape as X_test).
        y_test (array, optional): True labels corresponding to X_test samples. Defaults to None.
        num_samples (int): Number of samples to visualize (default: 2).
        eps (float, optional): Perturbation magnitude used to generate adversarial examples.
        plot_labels (list, optional): List of labels to filter which samples to plot. If None, plots all samples.
    """

    try:
        num_features = X_test.shape[2]
    except IndexError:
        num_features = 1

    plotted_samples = 0  # Counter to track how many samples have been plotted

    for sample in range(len(X_test)):
        if y_test is not None:
            label = y_test[sample]
            if plot_labels is not None and label not in plot_labels:
                continue  # Skip samples that don't match the specified labels
            label_info = f"(Label: {label})"
        else:
            label_info = ""

        if plotted_samples >= num_samples:
            break  # Stop if we've plotted the requested number of samples

        if num_features == 1:
            plt.figure(figsize=(10, 3))
            plt.plot(X_test[sample, :], label='Original', alpha=0.7)
            plt.plot(X_test_adv[sample, :], label=f'Adversarial - {eps}', alpha=0.7)
            plt.title(f'Sample {sample+1} {label_info} - Feature 1')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.legend()

        else:
            fig, axes = plt.subplots(num_features, 1, figsize=(10, num_features * 3))

            for i in range(num_features):
                axes[i].plot(X_test[sample, :, i], label='Original', alpha=0.7)
                axes[i].plot(X_test_adv[sample, :, i], label=f'Adversarial - {eps}', alpha=0.7)

                axes[i].set_title(f'Sample {sample+1} {label_info} - Feature {i+1}')
                axes[i].set_xlabel('Time Steps')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                plt.tight_layout()

        plt.show()
        plotted_samples += 1  # Increment plotted sample count

st.write('')

model_options_strongly = ['QML_strongly_5layers_100epochs_0.001lr_32batch_adamoptimizer',
 'QML_strongly_10layers_100epochs_0.001lr_32batch_adamoptimizer',
 'QML_strongly_25layers_100epochs_0.001lr_32batch_adamoptimizer',
 'QML_strongly_50layers_100epochs_0.001lr_32batch_adamoptimizer',
 'QML_strongly_100layers_100epochs_0.001lr_32batch_adamoptimizer']

list_of_models_strongly = [[path_to_data, model] for model in model_options_strongly]

for metric in metrics_to_plot:
    all_data = []
    # Loop through models and plot their performance for the current metric
    for model_selected in list_of_models_strongly:
        _, model_name = model_selected
        result_to_open = f'{adv_from[0]}/{adv_from[1]}/performance_of_{model_name}_on_{attack_type}_adv_from_{adv_from[1]}.json'
        
        # Print for debugging (optional)
        print(f"Opening: {result_to_open}")

        # Open the JSON file and extract metrics
        with open(result_to_open, "r") as f:
            metrics = json.load(f)

        eps_values = list(map(float, metrics['metrics'].keys()))  # Sorted epsilon values
        metric_values = [metrics['metrics'][str(eps)].get(metric, None) for eps in eps_values]

        # Collect data for plotting
        if None not in metric_values:
            for eps, value in zip(eps_values, metric_values):
                all_data.append({"Epsilon": eps, "Metric Value": value, "Model": model_name, "Metric": metric})

    # Convert collected data into a DataFrame for Plotly
    df = pd.DataFrame(all_data)

    # Create the Plotly figure for the current metric
    fig = px.line(
        df, x="Epsilon", y="Metric Value", color="Model", markers=True,
        title=f"{metric.title()} Performance",
        labels={"Epsilon": "Epsilon (ε)", "Metric Value": f"{metric.title()}"},
        hover_data={"Epsilon": True, "Metric Value": True, "Model": True, "Metric": True}
    )

    # Customize legend position
    fig.update_layout(
        legend=dict(
            orientation="h",  # Horizontal legend
            y=-0.2,  # Move legend down
            x=0.5,
            xanchor="center"
        ),
        yaxis=dict(range=[0, 1.1])  # Keep y-axis in range
    )
    st.plotly_chart(fig, use_container_width=True)