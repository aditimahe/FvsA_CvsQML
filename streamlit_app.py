import json
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import pandas as pd

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
    models_to_choose_from = ['QML_strongly_100layers_100epochs_0.001lr_64batch_adamoptimizer',
                            'QML_simplified_100layers_125epochs_0.001lr_32batch_adamoptimizer',
                            'CML_LSTM_0.01lr_50epochs_128batch_size_1layers_64hidden',
                            'CML_ConvNet_0.001lr_50epochs_128batch_size', 
                            'CML_ResNet34_0.001lr_50epochs_128batch_size']
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

st.write(list_of_models)
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

    # Footer text with dataset and attack details
    footer_text = f"""
    Dataset: {data_option}  
    Dimensionality Reduction: {data_details_option}  

    Adversarial Attack Type: {attack_type}  
    Adversarial Attack From {adv_from[1]}  
    """

    # Display footer text
    st.markdown(f"**{footer_text}**", unsafe_allow_html=True)

    # Show the Plotly chart
    st.plotly_chart(fig, use_container_width=True)




