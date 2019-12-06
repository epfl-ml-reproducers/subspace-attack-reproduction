import azureml.core
from azureml.core import Experiment, Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.train.dnn import PyTorch
from azureml.core.compute import RemoteCompute, ComputeTarget

azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 500 * 1000000

ws = Workspace.from_config()
experiment_name = 'ml-subspace-attack'
experiment = Experiment(workspace=ws, name=experiment_name)

run_local = RunConfiguration(framework='python')


script_params = {
    '--compare-gradients': 'True',
    '--n-images': '1'
}

cluster_name = 'local'

# Create the compute config 
compute_target_name = "attach-dsvm"
ip = ''
attach_config = RemoteCompute.attach_configuration(address = ip,
                                                  ssh_port=22,
                                                  username='azureuser',
                                                  password=None,
                                                  private_key_file='~/.ssh/azure_pem',
                                                  private_key_passphrase=None)

# Attach the compute
compute = ComputeTarget.attach(ws, compute_target_name, attach_config)

compute.wait_for_completion(show_output=True)


src = PyTorch(source_directory='.',
                      entry_script='subspace-attack.py',
                      compute_target=cluster_name,
                      use_gpu=True,
                      script_params=script_params)

run = experiment.submit(src)

print(experiment)

run.wait_for_completion(show_output=True)