# To add a new cell, type '#%%' test
# To add a new markdown cell, type '#%% [markdown]'
#%%
import os
os.chdir('C:\\Users\\flthibau\\Desktop\\contoso-workplace-safety')


#%%
run_id = 'hard-hat_1553713451380' ##'hard-hat_1553520422722'
run = Run(experiment=experiment, run_id=run_id)
scored_image = Image.open("./resources/After.jpg")
image__np = utils.load_image_into_numpy_array(scored_image)

#%% [markdown]
# ## Workplace safety with Azure Machine Learning
# 
#<img align="left" src="./score/samples/Before.jpg" width=750px style="margin-top:15px;margin-right:15px"/>
#<img align="left" src="./resources/After.jpg" width=750px/>
# 
#%% [markdown]
# ## Import open source Python libraries

#%%
import os
import requests
import utils
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import environment_definition

#%% [markdown]
# ## Import Azure Machine Learning Python SDK

#%%
from azureml.core import Workspace, Experiment
from azureml.core.model import Model
from azureml.core.run import Run
from azureml.widgets import RunDetails
from azureml.core.image import ContainerImage
from azureml.train.dnn import TensorFlow
from azureml.core.runconfig import AzureContainerRegistry, DockerEnvironment, EnvironmentDefinition, PythonEnvironment

from azureml.core.compute import AksCompute, ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.webservice import Webservice, AksWebservice

from azureml.core.image import Image
from azureml.contrib.iot import IotContainerImage

#%% [markdown]
# ## End to end: Build, Train and Deploy model
# 
# <img align="left" src="./resources/end2end.png" width=1200x />
#%% [markdown]
# ## Connect to Azure Machine Learning workspace
# 
# An Azure Machine Learning workspace is logically a container that has everything your team needs for machine learning
# 
# <img align="left" src="./resources/workspace.png" width=800x />

#%%
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

#%% [markdown]
# ## Collect and prepare training data
#%% [markdown]
# <HTML>
#     <TR>
#         <TD><img src="./images/IMG_1143.jpg" /></TD>
#         <TD><img src="./images/IMG_1195.jpg" /></TD>
#         <TD><img src="./images/IMG_1169.jpg" /></TD>
#         <TD><img src="./images/IMG_1172.jpg" /></TD>
#     </TR>
#     <TR>
#         <TD><img src="./images/IMG_1148.jpg" /></TD>
#         <TD><img src="./images/IMG_1183.jpg" /></TD>
#         <TD><img src="./images/IMG_1171.jpg" /></TD>
#         <TD><img src="./images/IMG_1152.jpg" /></TD>
#     </TR>
# </HTML>
#%% [markdown]
# ## Create Azure Machine Learning experiment

#%%
experiment_name = 'hard-hat'
experiment = Experiment(workspace=ws, name=experiment_name)

#%% [markdown]
# ## Create auto-scaling AML Compute GPU cluster

#%%
# Choose a name for your GPU cluster
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
cluster_name = "gpucluster"

# Verify that cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace = ws, name = cluster_name)
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                           min_nodes=0,
                                                           max_nodes=4)
    gpu_cluster = ComputeTarget.create(ws, cluster_name, compute_config)

gpu_cluster.wait_for_completion(show_output=True)

#%% [markdown]
# <img align="left" src="./resources/compute.png" width=800x />
#%% [markdown]
# ## Upload training data to Azure Machine Learning Data Store

#%%
# Prepare data
ds = ws.get_default_datastore()
ds.upload('./data')

#%% [markdown]
# ## Use prebuilt or your own custom docker container to train model for repeatability

#%%
from azureml.core import Experiment
from azureml.core import Run
from azureml.train.dnn import TensorFlow
from azureml.core.runconfig import AzureContainerRegistry, DockerEnvironment, EnvironmentDefinition, PythonEnvironment
env_def = EnvironmentDefinition()
env_def.docker = environment_definition.docker_config
env_def.python = environment_definition.python_config

print('Base docker image: ' + env_def.docker.base_image)
print('GPU Enabled: ' + str(env_def.docker.gpu_support))

#%% [markdown]
# ## Train model using TensorFlow estimator on the AMLCompute GPU cluster

#%%
script_params = {
    '--model_dir': './outputs',
    '--pipeline_config_path': './faster_rcnn_resnet101_hardhats.config'
}

tf_est = TensorFlow(source_directory = './train/src',
                    script_params=script_params,
                    compute_target=gpu_cluster,
                    entry_script='train.py',
                    inputs=[ds.as_download(path_on_compute='/data')],
                    environment_definition=env_def
                   )
run = experiment.submit(tf_est)


#%%
f = open("./train/src/train.py", "r") 
print(f.read())


#%%
#RunDetails(run).show() 
run.wait_for_completion(show_output=True)
#%% [markdown]
# ## Register model to track lineage, enable discoverability & reuse

#%%
# register the model for deployment
from azureml.core.model import Model
from azureml.core.run import Run
model = Model.register(model_path = "./models/frozen_inference_graph.pb",
                       model_name = "frozen_inference_graph.pb",
                       description = "Contoso Manufacturing model",
                       workspace = ws)


#%%
# Create a container image
from azureml.core.model import Model
from azureml.core.image import ContainerImage
os.chdir('C:\\Users\\flthibau\\Desktop\\contoso-workplace-safety\\score')

model = Model.list(ws, name='frozen_inference_graph.pb')[0]

image_config = ContainerImage.image_configuration(execution_script='score.py',
                                                  runtime='python',
                                                  conda_file='score.yml',
                                                  description='Object detection model')

image = ContainerImage.create(name='contosoimage',
                              models=[model],
                              image_config=image_config,
                              workspace=ws)

#%% [markdown]
# ## Create Azure Kubernetes Cluster

#%%
# Create an AKS cluster
from azureml.core.compute import AksCompute, ComputeTarget
aks_cluster_name = 'contoso-aks'

# Use the default configuration (can also provide parameters to customize)
prov_config = AksCompute.provisioning_configuration(location='eastus2')

# Create the cluster
aks_target = ComputeTarget.create(workspace=ws, 
                                  name=aks_cluster_name, 
                                  provisioning_configuration=prov_config)

aks_target.wait_for_completion(True)


#%%
# Deploy the model as a service

aks_service_name = 'contosoman'
aks_config = AksWebservice.deploy_configuration(collect_model_data=True, enable_app_insights=True)
aks_service = Webservice.deploy_from_image(workspace=ws, 
                                           name=aks_service_name,
                                           image=image,
                                           deployment_config=aks_config,
                                           deployment_target=aks_target)
aks_service

#%% [markdown]
# # Test the service

#%%
# Test the service
test_image = './score/samples/Before.jpg'
image = open(test_image, 'rb')
input_data = image.read()
image.close()

aks_service_name = 'contosoman2'
aks_service = AksWebservice(workspace=ws, name=aks_service_name)

auth = 'Bearer ' + aks_service.get_keys()[0]
uri = aks_service.scoring_uri

res = requests.post(url=uri,
                    data=input_data,
                    headers={'Authorization': auth, 'Content-Type': 'application/octet-stream'})

results = res.json()


#%%
# Show the results
image = Image.open(test_image)
image_np = utils.load_image_into_numpy_array(image)
category_index = utils.create_category_index_from_labelmap('./score/samples/label_map.pbtxt', use_display_name=True)

utils.visualize_boxes_and_labels_on_image_array(
    image_np,
    np.array(results['detection_boxes']),
    np.array(results['detection_classes']),
    np.array(results['detection_scores']),
    category_index,
    instance_masks=results.get('detection_masks'),
    use_normalized_coordinates=True,
    line_thickness=8)

plt.figure(figsize=(24, 16))
plt.imshow(image__np)

#%% [markdown]
# <img align="left" src="./resources/asos.png" width=1200x />
#%% [markdown]
# # Enabling the intelligent edge

#%%
# Build container image

image_config = IotContainerImage.image_configuration(
                                 architecture='arm32v7',
                                 execution_script='main.py',
                                 dependencies=['camera.py','iot.py','ipcprovider.py','utility.py','frame_iterators.py'],
                                 docker_file='Dockerfile',
                                 description='Object detection model (Edge)')

image = Image.create(name = 'contosoimage-edge',
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)

#%% [markdown]
# <img align="left" src="./resources/schneideroverview.png" width=1200x />
#%% [markdown]
# <img align="left" src="./resources/schneider.png" width=1200x />
#%% [markdown]
# # MLOps: use AML Pipelines to automate training and continuous model deployment
# 
# #### - Engie, an energy company, uses AML Pipelines to repeat the model training and deployment workflow for 1000's of models. 
# #### - AML pipelines are built on Aether, which is used to run millions of ML experiments per month in Microsoft
# #### - AML is integrated with Azure DevOps to include model training and deployment as part of your app / service DevOps lifecycle
# 
# <img align="left" src="./resources/MicrosoftTeams-image.png" width=1200x />
#%% [markdown]
# # How Engie uses Azure MLOps to train and deploy ML models continuously
# <img align="left" src="./resources/engie.png" width=1000x />
#%% [markdown]
# ### Engie's AML Pipelines train 1000's of anomaly detection pipelines for windmill turbines 
# 
# <img align="left" src="./resources/AMLPipelineEngie.png" width=1500x />
#%% [markdown]
# <img align="left" src="./resources/onnx.png" width=1200x />
#%% [markdown]
# <img align="left" src="./resources/powerbi.png" width=1200x />

#%%



