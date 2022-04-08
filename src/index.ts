import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, MainAreaWidget } from '@jupyterlab/apputils';
import { Widget } from '@lumino/widgets';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { requestAPI } from './handler';
import { ILauncher } from '@jupyterlab/launcher';

/* eslint-disable no-useless-escape */

async function generateNotebook(requestname: string, objBody: object, content: Widget) {
	try {
		const reply = await requestAPI<any>(
			requestname,
			{
			body: JSON.stringify(objBody),
			method: 'POST'
			}
		);
		console.log(reply)
		return reply
// ------------------------------------------------------------------------------------------------------------------------------- //
    } catch (reason) {
		console.error(
			`Error on POST /extension/`+requestname+` ${objBody}.\n${reason}`
		);
    } 
}

async function activate (app: JupyterFrontEnd, palette: ICommandPalette, launcher: ILauncher, settingRegistry: ISettingRegistry | null) {
	console.log('JupyterLab extension extension is activated!');
// setup and HTTP Request test; used to check if the server extension is enabled locally/ ob Binder
    if (settingRegistry) {
      settingRegistry
        .load(plugin.id)
        .then(settings => {
          console.log('extension settings loaded:', settings.composite);
        })
        .catch(reason => {
          console.error('Failed to load settings for extension.', reason);
        });
    }

    requestAPI<any>('get_example')
      .then(data => {
        console.log(data);
      })
      .catch(reason => {
        console.error(
          `The extension server extension appears to be missing.\n${reason}`
        );
      });
	  
	const dataToSend = { 'name': 'MLProvCodeGen1.0' };  
	try {
		const reply = await requestAPI<any>('post_example', {
		body: JSON.stringify(dataToSend),
		method: 'POST'
		});
		console.log(reply);
	} catch (reason) {
		console.error(
		`ERROR on post_example ${dataToSend}.\n${reason}`
		);
	}
// ------------------------------------------------------------------------------------------------------------------------------- //
  // Create a blank content widget inside of a MainAreaWidget
  const content = new Widget();
  const widget = new MainAreaWidget({ content });
  widget.id = 'MLProvCodeGen-jupyterlab';
  widget.title.label = 'MLProvCodeGen';
  widget.title.closable = true;
// ------------------------------------------------------------------------------------------------------------------------------- //
  // Button to reset widget
  const reset = document.createElement('div');
  content.node.appendChild(reset);
  reset.innerHTML = `
		<button id="reset" type="Button"> Reset this tab </button>  
		`;

  reset.addEventListener('click', event => {
    const nodeList = content.node.childNodes;
    console.log(nodeList);
    while (nodeList.length > 5) {
      nodeList[5].remove();
    }
  });
// ------------------------------------------------------------------------------------------------------------------------------- //
	const provenanceInputText = document.createElement('div');
	content.node.appendChild(provenanceInputText);
	provenanceInputText.id = 'provenanceInputText';
	provenanceInputText.innerHTML = '<b>Insert a MLProvCodeGen Provenance File:</b>';
	
	const provenanceInput = document.createElement('div');
	content.node.appendChild(provenanceInput);
	provenanceInput.innerHTML = `
		<input type="file" id="provenanceFileInput">
	`;
	
	provenanceInput.addEventListener('change', event => {
		let file = (<HTMLInputElement>event.target).files![0];
		const submitProvenanceFile = document.createElement('div');
        content.node.appendChild(submitProvenanceFile);
        submitProvenanceFile.innerHTML = `
						<button id="provenanceSubmit" name"provenanceSubmit" type="button"> Submit Provenance File </button>  
						`;
		
		submitProvenanceFile.addEventListener('click', async event => {
			console.log(file);
			
			let reader = new FileReader();
			reader.readAsText(file)
			reader.onload = async function() {
				//console.log(reader.result);
				var provenanceDataObj = JSON.parse(reader.result!.toString());
				console.log(provenanceDataObj); 
				console.log(provenanceDataObj.entity.experiment_info['experimentinfo:task_type']); 
				var taskName = provenanceDataObj.entity.experiment_info['experimentinfo:task_type'];
				var notebookPath = "('http://localhost:8888/lab/tree/extension/GeneratedNotebooks/"+provenanceDataObj.entity.experiment_info['experimentinfo:task_type']+".ipynb', 'MLProvCodeGen')";
				var openCall = `onclick="window.open`+notebookPath+`">`;
				console.log(openCall);
				
				const reply = await generateNotebook(taskName, provenanceDataObj, content)
				console.log(reply)
				if (reply['greetings'] === 'success') {
					const success_message = document.createElement('text');
					content.node.appendChild(success_message);
					success_message.id = 'successTextRight';
					success_message.textContent =
						'Your code has been generated successfully.';
							
					const notebook_open = document.createElement('div');
					content.node.appendChild(notebook_open);
					notebook_open.innerHTML = `
						<button id="openButtonRight" type="button" `+openCall+` Open Notebook </button>  
					`;
				}
			};
		}); // end of submitProvenanceFile event listener	
	}); // end of provenanceInput event listener
// ------------------------------------------------------------------------------------------------------------------------------- //
  const problemSelection = document.createElement('div');
  content.node.appendChild(problemSelection);
  problemSelection.innerHTML = `
	<form id="problemSelectionID" onsubmit="return false">
		<label for="exercise">Choose a problem to solve:</label>
		<select name="exercise" id="exercise">
			<option value="MulticlassClassification"> Multiclass Classification</option>
			<option value="ImageClassification"> Image Classification </option>
		</select>
	</form>	
  `;
  // ------------------------------------------------------------------------------------------------------------------------------- //
  // submit button for problem selection
  const problemSelectionButton = document.createElement('div');
  content.node.appendChild(problemSelectionButton);
  problemSelectionButton.innerHTML = `
		<button id="inputButton" type="button"> Submit </button> 
		`;

  problemSelectionButton.addEventListener('click', event => {
    const problemSubmit = (<HTMLSelectElement>(
      document.getElementById('exercise')
    )).value;
	
switch (problemSubmit) {
    case 'ImageClassification':
        const IC_model = document.createElement('div');
        content.node.appendChild(IC_model);
        IC_model.innerHTML = `
						<form action="/action_page.php">
							<label for="model">Which model do you want to use?</label>
							<select name="model" id="model">
								<option value="resnet18"> resnet18 </option>
								<option value="densenet161"> densenet161 </option>
								<option value="vgg16"> vgg16 </option>
							</select>
						</form>	
						`;
        const IC_classes = document.createElement('div');
        content.node.appendChild(IC_classes);
		IC_classes.innerHTML = `
						<form action="/action_page.php">
							<label for="quantity">How many classes/output units?</label>
							<input type="number" id="quantity" name="quantity" value="1000"> 
							Default: 1000 classes for training on ImageNet
						</form>
						`;
		const IC_pretrained = document.createElement('div');
		content.node.appendChild(IC_pretrained);
		IC_pretrained.innerHTML = `
						<form>
							<input type="checkbox" id="preTrainedModel" name="preTrainedModel" value="preTrainedModel">
							<label for="preTrainedModel"> Do you want to use a pre trained model?</label><br>
						</form>
						`;
		const IC_dataFormat = document.createElement('div');
		content.node.appendChild(IC_dataFormat);
		IC_dataFormat.innerHTML = `
						<form action="/action_page.php">
							<label for="data">Which data do you want to use?</label>
							<select name="data" id="data">
								<option value="Public dataset"> Public dataset </option>
								<option value="Numpy arrays"> Numpy arrays </option>
								<option value="Image files"> Image files </option>
							</select>
						</form>	
						`;
		const IC_dataset = document.createElement('div');
		content.node.appendChild(IC_dataset);
		IC_dataset.innerHTML = `
						<form action="/action_page.php">
						<label for="dataSelection">Which one?:</label>
						<select name="dataSelection" id="dataSelection">
							<option value="MNIST"> MNIST </option>
							<option value="FashionMNIST"> FashionMNIST </option>
							<option value="CIFAR10"> CIFAR10 </option>
						</select>
						</form>	
						`;
		const IC_useGPU = document.createElement('div');
		content.node.appendChild(IC_useGPU);
		IC_useGPU.innerHTML = `
						<form>
							<input type="checkbox" id="useGPU" name="useGPU" value="useGPU" checked>
							<label for="useGPU"> Use GPU if available? </label><br>
						</form>
						`;
		const seed = document.createElement('div');
		content.node.appendChild(seed);
		seed.innerHTML = `
						<form action="/action_page.php">
							<label for="seed"> Random Seed</label>
							<input type="number" id="seed" name="seed" value="2">
						</form>
						`; 
		const IC_checkpoint = document.createElement('div');
		content.node.appendChild(IC_checkpoint);
		IC_checkpoint.innerHTML = `
						<form>
							<input type="checkbox" id="modelCheckpoint" name="modelCheckpoint" value="modelCheckpoint">
							<label for="modelCheckpoint"> Save model checkpoint each epoch?</label><br>
						</form>
						`;          
		const IC_lossFunction = document.createElement('div');
		content.node.appendChild(IC_lossFunction);
		IC_lossFunction.innerHTML = `
						<form action="/action_page.php">
							<label for="lossFunc"> Loss function</label>
							<select name="lossFunc" id="lossFunc">
								<option value="CrossEntropyLoss"> CrossEntropyLoss </option>
								<option value="BCEWithLogitsLoss"> BCEWithLogitsLoss </option>
							</select>
						</form>	
						`;
		const IC_optimizer = document.createElement('div');
		content.node.appendChild(IC_optimizer);
		IC_optimizer.innerHTML = `
						<form action="/action_page.php">
							<label for="optimizer"> Optimizer </label>
							<select name="optimizer" id="optimizer">
								<option value="Adam"> Adam </option>
								<option value="Adadelta"> Adadelta </option>
								<option value="Adagrad"> Adagrad </option>
								<option value="Adamax"> Adamax </option>
								<option value="RMSprop"> RMSprop </option>
								<option value="SGD"> SGD </option>
							</select>
						</form>	
						`;
		const IC_learningRate = document.createElement('div');
		content.node.appendChild(IC_learningRate);
		IC_learningRate.innerHTML = `
						<form action="/action_page.php">
							<label for="rate"> Learning rate</label>
							<input type="number" id="rate" name="rate" value="0.001">
						</form>
						`;
		const IC_batchSize = document.createElement('div');
		content.node.appendChild(IC_batchSize);
		IC_batchSize.innerHTML = `
						<form action="/action_page.php">
							<label for="batches"> Batch Size</label>
							<input type="number" id="batches" name="batches" value="128">
						</form>
						`;
		const IC_epochs = document.createElement('div');
		content.node.appendChild(IC_epochs);
		IC_epochs.innerHTML = `
						<form action="/action_page.php">
							<label for="epochs">How many epochs?</label>
							<input type="number" id="epochs" name="epochs" value="3">
						</form>
						`;
		const IC_printProgress = document.createElement('div');
		content.node.appendChild(IC_printProgress);
		IC_printProgress.innerHTML = `
						<form action="/action_page.php">
							<label for="printProgress"> Print progress every ... batches</label>
							<input type="number" id="printProgress" name="printProgress" value="1">
						</form>
						`;
		const IC_logging = document.createElement('div');
		content.node.appendChild(IC_logging);
		IC_logging.innerHTML = `
						<form action="/action_page.php">
							<label for="logs"> How to log metrics </label>
							<select name="logs" id="logs">
								<option value="notAtAll"> Not at all </option>
								<option value="Tensorboard"> Tensorboard </option>
								<option value="Aim"> Aim </option>
								<option value="Weights & Biases"> weightsAndBiases </option>
								<option value="comet.ml"> comet.ml </option>
							</select>
						</form>	
						`;
		const submitButtonIC = document.createElement('div');
		content.node.appendChild(submitButtonIC);
		submitButtonIC.innerHTML = `
						<button id="inputButton" type="button"> Submit your values </button>  
						`;
// ------------------------------------------------------------------------------------------------------------------------------- //
        submitButtonIC.addEventListener('click', async event => {
            const exerciseValue = (<HTMLSelectElement>(
              document.getElementById('exercise')
            )).value;
            const modelValue = (<HTMLSelectElement>(
              document.getElementById('model')
            )).value;
            const dataValue = (<HTMLSelectElement>(
              document.getElementById('data')
            )).value;
            const dataSelectionValue = (<HTMLSelectElement>(
              document.getElementById('dataSelection')
            )).value;
            const lossFuncValue = (<HTMLSelectElement>(
              document.getElementById('lossFunc')
            )).value;
            const optimizerValue = (<HTMLSelectElement>(
              document.getElementById('optimizer')
            )).value;
            const logsValue = (<HTMLSelectElement>(
              document.getElementById('logs')
            )).value;
            const quantityValue = (<HTMLInputElement>(
              document.getElementById('quantity')
            )).value;
            const rateValue = (<HTMLInputElement>(
              document.getElementById('rate')
            )).value;
            const batchesValue = (<HTMLInputElement>(
              document.getElementById('batches')
            )).value;
            const epochsValue = (<HTMLInputElement>(
              document.getElementById('epochs')
            )).value;
            const printProgressValue = (<HTMLInputElement>(
              document.getElementById('printProgress')
            )).value;
			const seedValue = (<HTMLInputElement>(
              document.getElementById('seed')
            )).value;
            let preTrainedModelValue = 3;
            let useGPUValue = 3;
            let modelCheckpointValue = 3;
            if (
              (<HTMLInputElement>document.getElementById('preTrainedModel'))
                .checked
            ) {
              preTrainedModelValue = 1;
            } else {
              preTrainedModelValue = 0;
            }

            if ((<HTMLInputElement>document.getElementById('useGPU')).checked) {
              useGPUValue = 1;
            } else {
              useGPUValue = 0;
            }

            if (
              (<HTMLInputElement>document.getElementById('modelCheckpoint'))
                .checked
            ) {
              modelCheckpointValue = 1;
            } else {
              modelCheckpointValue = 0;
            }
// ------------------------------------------------------------------------------------------------------------------------------- //
            const objBody = {
				exercise: exerciseValue,
				'entity':{
					'data_ingestion': {
						'dataingestion:data_format': dataValue,
						'dataingestion:dataset_id': dataSelectionValue
					},
					'model_parameters': {
						'modelparameters:model_name': modelValue,
						'modelparameters:pretrained': {
							'$': preTrainedModelValue,
							'type': typeof(preTrainedModelValue),
						},
						'modelparameters:gpu_enable': {
							'$': useGPUValue,
							'type': typeof(useGPUValue),
						},
						'modelparameters:num_classes': {
							'$': quantityValue, 
							'type': typeof(quantityValue),
						},
						'modelparameters:save_checkpoint': {
							'$': modelCheckpointValue,
							'type': typeof(modelCheckpointValue),
						},
						'modelparameters:loss_function': lossFuncValue,
						'modelparameters:optimizer': optimizerValue,
						'modelparameters:optimizer_learning_rate': {
							'$': rateValue,
							'type': typeof(rateValue)
						},
					},
					'training': {
						'training:batch_size': {
							'$': batchesValue,
							'type': typeof(batchesValue),
						},
						'training:epochs': {
							'$': epochsValue,
							'type': typeof(epochsValue),
						},
						'training:print_progress': {
							'$': printProgressValue,
							'type': typeof(printProgressValue),
						},
						'training:seed': {
							'$': seedValue,
							'type': typeof(seedValue)
						}
					},
					'visualization_tool':{
						'tool' : logsValue
					},
				}
			};
			var method = 'ImageClassification_pytorch'
			const reply = await generateNotebook(method, objBody, content)
			console.log(reply);
			
			if (reply['greetings'] === 'success') {
				const success_message = document.createElement('text');
				content.node.appendChild(success_message);
				success_message.textContent =
				'Your Code has been generated successfully. Press the button below to open it.';
				
				const notebook_open = document.createElement('div');
				content.node.appendChild(notebook_open);
				notebook_open.innerHTML = `
					<button id="inputButton" type="button" onclick="window.open('http://localhost:8888/lab/workspaces/auto-y/tree/extension/GeneratedNotebooks/ImageClassification_PyTorch.ipynb', 'MLProvCodeGen')"> Open Notebook </button>  
					`;
			}
        }); // end of submitButton event listener
	break;
case 'MulticlassClassification':
      // UI Inputs
        const MC_data_header = document.createElement('div');
        content.node.appendChild(MC_data_header);
        MC_data_header.innerHTML = `
						<b><u> Data Settings</u></b>
					`;

        const MC_dataset = document.createElement('div');
        content.node.appendChild(MC_dataset);
        MC_dataset.innerHTML = `
						<form action="/action_page.php">
							<label for="dataset">Which dataset do you want to use?</label>
							<select name="dataset" id="dataset">
								<option value="Iris"> Iris </option>
								<option value="Spiral"> Spiral </option>
								<option value="Aggregation"> Aggregation </option>
								<option value="R15"> R15 </option>
								<option value="User"> Use your own Data (.csv) </option>
							</select>
						</form>	
						`;
        const MC_random_seed = document.createElement('div');
        content.node.appendChild(MC_random_seed);
        MC_random_seed.innerHTML = `
						<form action="/action_page.php">
							<label for="random_seed">Random Seed for data Segregation (default: 2)</label>
							<input type="number" id="random_seed" name="random_seed" value="2">
						</form>
						`;

        const MC_test_split = document.createElement('div');
        content.node.appendChild(MC_test_split);
        MC_test_split.innerHTML = `
						<form action="/action_page.php">
							<label for="test_split">Testing data split (default: 0.2)</label>
							<input type="number" id="test_split" name="test_split" value="0.2">
						</form>
						`;

        const MC_preprocessing_text = document.createElement('div');
        content.node.appendChild(MC_preprocessing_text);
        MC_preprocessing_text.innerHTML = `
						<label> <i>preprocessing: MinMaxScaler</i></label>
					`;

        const MC_model_header = document.createElement('div');
        content.node.appendChild(MC_model_header);
        MC_model_header.innerHTML = `
						<b><u> Model Settings</u></b>
					`;

        const MC_activation_func = document.createElement('div');
        content.node.appendChild(MC_activation_func);
        MC_activation_func.innerHTML = `
						<form action="/action_page.php">
							<label for="activation_func">Activation function:</label>
							<select name="activation_func" id="activation_func">
								<option value="F.softmax(self.layer3(x), dim=1)"> Softmax </option>
								<option value="torch.sigmoid(self.layer3(x))"> Sigmoid </option>
								<option value="torch.tanh(self.layer3(x))"> Tanh </option>
							</select>
						</form>	
						`;

        const MC_neuron_number = document.createElement('div');
        content.node.appendChild(MC_neuron_number);
        MC_neuron_number.innerHTML = `
						<form action="/action_page.php">
							<label for="neuron_number">How many Neurons per linear layer? (Input and output neurons are separate) </label>
							<input type="number" id="neuron_number" name="neuron_number" value="50">
						</form>
						`;

        const MC_epochs = document.createElement('div');
        content.node.appendChild(MC_epochs);
        MC_epochs.innerHTML = `
						<form action="/action_page.php">
							<label for="epochs">How many Epochs?</label>
							<input type="number" id="epochs" name="epochs" value="100">
						</form>
						`;

        const MC_optimizer = document.createElement('div');
        content.node.appendChild(MC_optimizer);
        MC_optimizer.innerHTML = `
						<form action="/action_page.php">
							<label for="optimizer"> Optimizer </label>
							<select name="optimizer" id="optimizer">
								<option value="torch.optim.Adam("> Adam </option>
								<option value="torch.optim.Adadelta("> Adadelta </option>
								<option value="torch.optim.Adagrad("> Adagrad </option>
								<option value="torch.optim.Adamax("> Adamax </option>
								<option value="torch.optim.RMSprop("> RMSprop </option>
								<option value="torch.optim.SGD("> SGD </option>
							</select>
						</form>	
						`;
        const MC_default_lr = document.createElement('div');
        content.node.appendChild(MC_default_lr);
        MC_default_lr.innerHTML = `
						<form>
							<input type="checkbox" id="default" name="default" value="default" checked>
							<label for="default"> Use optimizers default learning rate? </label><br>
						</form>
						`;

        const MC_lr = document.createElement('div');
        content.node.appendChild(MC_lr);
        MC_lr.innerHTML = `
						<form action="/action_page.php">
							<label for="rate"> Learning rate</label>
							<input type="number" id="rate" name="rate" value="0.001">
						</form>
						`;
        const MC_loss = document.createElement('div');
        content.node.appendChild(MC_loss);
        MC_loss.innerHTML = `
						<form action="/action_page.php">
							<label for="lossFunc"> Loss function</label>
							<select name="lossFunc" id="lossFunc">
								<option value="nn.CrossEntropyLoss()"> CrossEntropyLoss </option>
								<option value="nn.NLLLoss()"> NLLLoss </option>
								<option value="nn.MultiMarginLoss()"> MultiMarginLoss </option>
							</select>
						</form>	
						`;
        const MC_use_gpu = document.createElement('div');
        content.node.appendChild(MC_use_gpu);
        MC_use_gpu.innerHTML = `
						<form>
							<input type="checkbox" id="use_gpu" name="use_gpu" value="use_gpu" checked>
							<label for="use_gpu"> Use GPU if available? </label><br>
						</form>
						`;

        const MC_submitButton = document.createElement('div');
        content.node.appendChild(MC_submitButton);
        MC_submitButton.innerHTML = `
						<button id="inputButton" type="button"> Submit your values </button>  
						`;
        MC_submitButton.addEventListener('click', async event => {
// ------------------------------------------------------------------------------------------------------------------------------- //
			// Get Variables
			const exercise = (<HTMLSelectElement>(
				document.getElementById('exercise')
			)).value;
			const dataset = (<HTMLSelectElement>(
				document.getElementById('dataset')
			)).value;
			const activation_func = (<HTMLSelectElement>(
				document.getElementById('activation_func')
			)).value;
			const optimizer = (<HTMLSelectElement>(
				document.getElementById('optimizer')
			)).value;
			const loss_func = (<HTMLSelectElement>(
				document.getElementById('lossFunc')
			)).value;

			const test_split = (<HTMLInputElement>(
				document.getElementById('test_split')
			)).value;
			const neuron_number = (<HTMLInputElement>(
				document.getElementById('neuron_number')
			)).value;
			const epochs = (<HTMLInputElement>document.getElementById('epochs'))
				.value;
			const lr = (<HTMLInputElement>document.getElementById('rate')).value;
			const random_seed = (<HTMLInputElement>(
				document.getElementById('random_seed')
			)).value;

			let defaultValue, use_gpu;
			if ((<HTMLInputElement>document.getElementById('default')).checked) {
				defaultValue = 1;
			} else {
				defaultValue = 0;
			}

			if ((<HTMLInputElement>document.getElementById('use_gpu')).checked) {
				use_gpu = 1;
			} else {
				use_gpu = 0;
			}
// ------------------------------------------------------------------------------------------------------------------------------- //
			// convert variables into JSON/ input Object
			const objBody = {
				exercise: exercise,
				'entity':{
					'data_ingestion': {
						'dataingestion:dataset_id': dataset
					},			
					'data_segregation': {
						'datasegregation:test_size':{
							'$': test_split,
							'type' : typeof(test_split),
						},
						'datasegregation:random_state': {
							'$': random_seed,
							'type': typeof(random_seed),
						},
					},
					'model_parameters': {
						'modelparameters:gpu_enable': {
							'$':use_gpu,
							'type':typeof(use_gpu),
						},
						'modelparameters:neuron_number': {
							'$':neuron_number,
							'type':typeof(neuron_number),
						},
						'modelparameters:loss_function': loss_func,
						'modelparameters:optimizer': optimizer,
						'modelparameters:optimizer_default_learning_rate':{
							'$':defaultValue,
							'type': typeof(defaultValue),
						},
						'modelparameters:optimizer_learning_rate':{
							'$': lr,
							'type': typeof(lr),
						},
						'modelparameters:activation_function': activation_func
					},
					'training': {
						'training:epochs': {
							'$':epochs,
							'type':typeof(epochs),
						}
					},
				}
			};
			var method = 'MulticlassClassification'
			const reply = await generateNotebook(method, objBody, content)
			console.log(reply);
			if (reply["greetings"] === 'success') {
				var path = window.location.href + '/tree/GeneratedNotebooks/MulticlassClassification.ipynb'
				console.log(path)
				const success_message = document.createElement('text');
				content.node.appendChild(success_message);
				success_message.textContent =
					'Your Code has been generated successfully. Press the button below to open it.';

				const notebook_open = document.createElement('div');
				content.node.appendChild(notebook_open);
				/*notebook_open.innerHTML = `
									<button id="inputButton" type="button" onclick="window.open('http://localhost:8888/lab/tree/extension/GeneratedNotebooks/MulticlassClassification.ipynb', 'MLProvCodeGen')"> Open Notebook </button>  
									`;*/
				notebook_open.innerHTML = `
									<button id="inputButton" type="button" onclick="window.open(`+path+`, 'MLProvCodeGen')"> Open Notebook </button>  
									`;					
			}
        }); // end of SubmitButton event listener
		console.log(window.location.href)
    break;
	} // end switch
	});// end on the problemSelectionButton event listener
// ------------------------------------------------------------------------------------------------------------------------------- //
	// Add an application command
	const command = 'codegenerator:open';
	app.commands.addCommand(command, {
		label: 'MLProvCodeGen',
		execute: () => {
		if (!widget.isAttached) {
			// Attach content to the main work area if it's not there
			app.shell.add(widget, 'main');
		}
		// Activate the widget
		app.shell.activateById(widget.id);
		}
	});
	// Add the command to the palette.
	palette.addItem({ command, category: 'MLProvCodeGen' });	
	launcher.add({ command, category: 'Other', rank: 0 });
}
// ------------------------------------------------------------------------------------------------------------------------------- //  
// Main
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'extension:plugin',
  autoStart: true,
  requires: [ICommandPalette, ILauncher],
  optional: [ISettingRegistry],
  activate: activate
}
    

export default plugin;