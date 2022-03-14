import nbformat as nbf
import json
import sys
from jinja2 import Environment, FileSystemLoader

def app(code):
    nb = nbf.v4.new_notebook()

    text = """\
    # My first automatic Jupyter Notebook
    This is an auto-generated notebook."""

    code2 = """\
    %pylab inline
    hist(normal(size=2000), bins=50);"""

    nb['cells'] = [nbf.v4.new_markdown_cell(text),
                    nbf.v4.new_code_cell(code2)]


    cellText = str(code)
    nb['cells'].append(nbf.v4.new_markdown_cell(cellText))



    nbf.write(nb, 'GeneratedNotebooks/userInputNotebook.ipynb')

    return code

def IC_pytorch(user_inputs):
    nb = nbf.v4.new_notebook()

    text = """
# Image Classification

Building a machine learning model to solve Image Classification using the PyTorch framework.<br>
Image Classification is one of the basic pattern recognition exercises. <br>
Using Image files as its input, a model trained for Image classification will split a set of images into a given number of classes. <br>
<br>
This Notebook has been generated automatically using the JupyterLab extension ***MLProvCodeGen***.
<br>
The original Source Code is from this application https://github.com/jrieke/traingenerator <br>
Made by: https://www.jrieke.com/ Twitter: https://twitter.com/jrieke
"""
    nb['cells'] = [nbf.v4.new_markdown_cell(text)]

    file_loader = FileSystemLoader('jinjaTemplates/IC_pytorch')
    env = Environment(loader=file_loader, trim_blocks=True, lstrip_blocks=True)
# GET VARIABLES
    problemName = user_inputs['exercise']
    selectedFramework = user_inputs['framework']
    visualization_tool = user_inputs['visualization_tool']
    notebook = True
    data_format = user_inputs['data']
    checkpoint = user_inputs['model_checkpoint']
    lr = user_inputs['rate']
    batch_size = user_inputs['batches']
    num_epochs = user_inputs['epochs']
    print_every = user_inputs['print_progress']
    gpu = user_inputs['use_GPU']
    dataset = user_inputs['data_selection']
    model_func = user_inputs['model']
    pretrained = user_inputs['pre_trained_model']
    num_classes = user_inputs['quantity']
    loss = user_inputs['loss_function']
    optimizer = user_inputs['optimizer']
#-----------------------------------------------------------------------------
    # installs
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Installs
Install required packages before running"""))
    template = env.get_template('installs.jinja')
    output = template.render(visualization_tool = visualization_tool)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    # imports
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Imports"""))
    template = env.get_template('imports.jinja')
    output = template.render(visualization_tool = visualization_tool, data_format = data_format, checkpoint = checkpoint)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #preSetup
    if data_format == "Numpy arrays" or data_format == "Image files":
        nb['cells'].append(nbf.v4.new_markdown_cell("""### preSetup"""))
        template = env.get_template('preSetup.jinja')
        output = template.render(data_format = data_format)
        nb['cells'].append(nbf.v4.new_code_cell(output))

    #Setup
    # TODO: each logging method needs its own variables
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Setup"""))
    template = env.get_template('setup.jinja')
    output = template.render(data_format = data_format, lr = lr, batch_size = batch_size, num_epochs = num_epochs, visualization_tool = visualization_tool, gpu = gpu, print_every = print_every)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Datasets and Preprocessing
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Datasets and Preprocessing"""))
    template = env.get_template('datasetPreprocessing.jinja')
    output = template.render(data_format = data_format, dataset = dataset, pretrained = pretrained, gpu = gpu)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Model
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Model"""))
    template = env.get_template('model.jinja')
    output = template.render(model_func = model_func, pretrained = pretrained, num_classes = num_classes, loss = loss, optimizer = optimizer, visualization_tool = visualization_tool)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Training
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Training"""))
    template = env.get_template('training.jinja')
    output = template.render(visualization_tool = visualization_tool)
    nb['cells'].append(nbf.v4.new_code_cell(output))


    nbf.write(nb, 'GeneratedNotebooks/ImageClassification_PyTorch.ipynb')
    reply = {"greetings": "success"}
    return reply

def IC_scikit(user_inputs):
    nb = nbf.v4.new_notebook()

    text = """
# Image Classification

Building a machine learning model to solve Image Classification using the scikit-learn framework.<br>
Image Classification is one of the basic pattern recognition exercises. <br>
Using Image files as its input, a model trained for Image classification will split a set of images into a given number of classes. <br>
<br>
This Notebook has been generated automatically using the JupyterLab extension ***MLProvCodeGen***.
<br>

The original Source Code is from this application https://github.com/jrieke/traingenerator <br>
Made by: https://www.jrieke.com/ Twitter: https://twitter.com/jrieke
        """
    nb['cells'] = [nbf.v4.new_markdown_cell(text)]

    file_loader = FileSystemLoader('jinjaTemplates/IC_scikit')
    env = Environment(loader=file_loader, trim_blocks=True, lstrip_blocks=True)
    # TODO: scikit template with 5 variables
    #   model variable might be really weird
    #   fix Headers since i deleleted them from pytorch AND scikit-learn
    #   scikit image files needs a small addition
    #   pytorch needs alot of prompts

    template = env.get_template('variableConversion.jinja')

    model_func = user_inputs['model_func']
    data_format = user_inputs['data_format']
    scale_mean_std = user_inputs['scale_mean_std']
    visualization_tool = user_inputs['visualization_tool']
    resize_pixels = 28
    crop_pixels = 28
    comet_api_key = user_inputs['cometAPIKey']
    comet_project = user_inputs['cometName']

    # installs
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Installs
Install required packages before running"""))
    template = env.get_template('installs.jinja')
    output = template.render(visualization_tool = visualization_tool)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    # imports
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Imports"""))
    template = env.get_template('imports.jinja')
    output = template.render(visualization_tool = visualization_tool, model_func = model_func, data_format = data_format)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    # setup
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Setup"""))
    template = env.get_template('setup.jinja')
    output = template.render(data_format = data_format, visualization_tool = visualization_tool, comet_api_key = comet_api_key, comet_project = comet_project)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    # preprocessing
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Preprocessing"""))
    template = env.get_template('preprocessing.jinja')
    output = template.render(scale_mean_std = scale_mean_std, data_format = data_format, resize_pixels = resize_pixels, crop_pixels = crop_pixels)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    # Model
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Model"""))
    template = env.get_template('model.jinja')
    output = template.render(model_func = model_func)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    # Training
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Training"""))
    template = env.get_template('training.jinja')
    output = template.render(visualization_tool = visualization_tool)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    # Evaluation
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Evaluation"""))
    template = env.get_template('evaluation.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    nbf.write(nb, 'GeneratedNotebooks/ImageClassification_Scikit.ipynb')

    reply = {"greetings": "success"}

    return reply

def Clustering_scikit(user_inputs):

    model_func = user_inputs['model_func']
    data_format = user_inputs['data_format']
    preprocessing = user_inputs['preprocessing']

    nb = nbf.v4.new_notebook()

    text = """This is an auto-generated notebook."""
    nb['cells'] = [nbf.v4.new_markdown_cell(text)]

    file_loader = FileSystemLoader('jinjaTemplates/scikitClustering')
    env = Environment(loader=file_loader, trim_blocks=True, lstrip_blocks=True)
    # add a cell for each step
    #pip installs:
    template = env.get_template('pipInstalls.jinja')
    output = template.render(model_func = model_func)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #imports
    template = env.get_template('imports.jinja')
    output = template.render(model_func = model_func)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #load_data
    template = env.get_template('data.jinja')
    output = template.render(model_func = model_func, data_format = data_format)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #plot data initially
    template = env.get_template('plotInitial.jinja')
    output = template.render(model_func = model_func, data_format = data_format)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    if preprocessing == True:
        #preprocessing
        template = env.get_template('preprocessing.jinja')
        output = template.render(model_func = model_func, data_format = data_format)
        nb['cells'].append(nbf.v4.new_code_cell(output))

    #main func
    template = env.get_template('mainfunc.jinja')
    output = template.render(model_func = model_func, data_format = data_format)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #show predictions
    template = env.get_template('showPredictions.jinja')
    output = template.render(model_func = model_func)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #show predictions
    template = env.get_template('plotResults.jinja')
    output = template.render(model_func = model_func, data_format = data_format)
    nb['cells'].append(nbf.v4.new_code_cell(output))



    #template = env.get_template('Clustering_scikit.jinja')




    #output = template.render(model_func = model_func, data_format = data_format)

    #nb['cells'].append(nbf.v4.new_code_cell(output))
    nbf.write(nb, 'GeneratedNotebooks/userInputNotebook.ipynb')

    reply = {"greetings": "success"}

    return reply


def MS_scikit(user_inputs):
    nb = nbf.v4.new_notebook()

    nb['cells'] = []

    file_loader = FileSystemLoader('jinjaTemplates/MS_scikit')
    env = Environment(loader=file_loader, trim_blocks=True, lstrip_blocks=True)
# GET VARIABLES
    problemName = user_inputs['exercise']
    selectedFramework = user_inputs['framework']
    visualization_tool = user_inputs['visualization_tool']
    notebook = True
    data_format = user_inputs['data']
    checkpoint = user_inputs['model_checkpoint']
    lr = user_inputs['rate']
    batch_size = user_inputs['batches']
    num_epochs = user_inputs['epochs']
    print_every = user_inputs['print_progress']
    gpu = user_inputs['use_GPU']
    dataset = user_inputs['data_selection']
    model_func = user_inputs['model']
    pretrained = user_inputs['pre_trained_model']
    num_classes = user_inputs['quantity']
    loss = user_inputs['loss_function']
    optimizer = user_inputs['optimizer']
#-----------------------------------------------------------------------------
    # installs
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Installs
Install required packages before running"""))
    template = env.get_template('installs.jinja')
    output = template.render(visualization_tool = visualization_tool, data_format = data_format)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    # imports
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Imports"""))
    template = env.get_template('imports.jinja')
    output = template.render(visualization_tool = visualization_tool, checkpoint = checkpoint, model_func = model_func)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #preSetup
    nb['cells'].append(nbf.v4.new_markdown_cell("""### preSetup"""))
    template = env.get_template('preSetup.jinja')
    output = template.render(data_format = data_format)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Setup
    # TODO: each logging method needs its own variables
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Setup"""))
    template = env.get_template('setup.jinja')
    output = template.render(data_format = data_format, dataset = dataset, lr = lr, batch_size = batch_size, num_epochs = num_epochs, visualization_tool = visualization_tool, checkpoint = checkpoint, print_every = print_every, gpu = gpu)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Datasets and Preprocessing
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Datasets and Preprocessing"""))
    template = env.get_template('datasetPreprocessing.jinja')
    output = template.render(data_format = data_format, dataset = dataset, pretrained = pretrained, gpu = gpu)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Model
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Model"""))
    template = env.get_template('model.jinja')
    output = template.render(model_func = model_func, pretrained = pretrained, num_classes = num_classes, loss = loss, optimizer = optimizer, visualization_tool = visualization_tool)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Training
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Training"""))
    template = env.get_template('training.jinja')
    output = template.render(visualization_tool = visualization_tool)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    nbf.write(nb, 'GeneratedNotebooks/ModelSelection_scikit.ipynb')

    reply = {"greetings": "success"}

    return reply

def MulticlassClassification(user_inputs):
    #_(sys.executable, "cd", "..")
    nb = nbf.v4.new_notebook()
    
    text = """
# Multiclass Classification

Building a neural network to solve a multiclass classification exercise using the PyTorch framework.<br>
Classification is one of the basic machine learning exercises. A trained model aims to predict the class of an input unit with high accuracy. <br>
This neural network uses supervised learning, meaning that the input datasets also provide target labels to train the model with. <br>
<br>
This Notebook has been generated automatically using the JupyterLab extension ***MLProvCodeGen***.
<br>

Original Source Code and inspiration from this article https://janakiev.com/blog/pytorch-iris/ <br>
Original author: N. Janakiev https://github.com/njanakiev Twitter: https://twitter.com/njanakiev
        """

    nb['cells'] = [nbf.v4.new_markdown_cell(text)]

    file_loader = FileSystemLoader('jinjaTemplates/MulticlassClassification')
    env = Environment(loader=file_loader, trim_blocks=True, lstrip_blocks=True)
    ###Extract Variables from user_inputs
    dataset = user_inputs['data_ingestion']['dataset_id']
    random_seed = user_inputs['data_segregation']['random_state']
    test_split = user_inputs['data_segregation']['test_size']
    #activation_func = user_inputs['activation_func']
    activation_func = user_inputs['model_parameters']['activation_function']
    neuron_number = user_inputs['model_parameters']['neuron_number']
    #optimizer = user_inputs['optimizer']
    optimizer = user_inputs['model_parameters']['optimizer']
    #loss_func = user_inputs['loss_func']
    loss_func = user_inputs['model_parameters']['loss_function']
    #epochs = user_inputs['epochs']
    epochs = user_inputs['training']['epochs']
    #lr = user_inputs['lr']
    lr = user_inputs['model_parameters']['optimizer_learning_rate']
    #use_gpu = user_inputs['use_gpu']
    use_gpu = user_inputs['model_parameters']['gpu_enable']
    #default = user_inputs['default']
    default = user_inputs['model_parameters']['optimizer_default_learning_rate']


    #installs
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Installs
Install required packages before running"""))
    template = env.get_template('installs.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #imports
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Imports"""))
    template = env.get_template('imports.jinja')
    output = template.render(dataset = dataset)
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #provenance (experiment, hardware, packages
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Provenance Data """))
    template = env.get_template('provenanceInit.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #dataIngestion
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Ingestion"""))
    template = env.get_template('dataIngestion.jinja')
    output = template.render(dataset = dataset)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #dataPreperation
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Preperation"""))
    template = env.get_template('dataPreperation.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #dataSegregation
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Segregation"""))
    template = env.get_template('dataSegregation.jinja')
    output = template.render(random_seed = random_seed, test_split = test_split)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #dataVisualization
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Data Visualization"""))
    template = env.get_template('dataVisualization.jinja')
    output = template.render(dataset = dataset)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #model
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Model"""))
    template = env.get_template('model.jinja')
    output = template.render(optimizer = optimizer,default = default, lr = lr, loss_func = loss_func, activation_func = activation_func, use_gpu = use_gpu, dataset = dataset, neuron_number = neuron_number)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #training
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Training"""))
    template = env.get_template('training.jinja')
    output = template.render(epochs = epochs, dataset = dataset)
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #evaluation
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Evaluation"""))
    template = env.get_template('evaluation.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #ROC
    nb['cells'].append(nbf.v4.new_markdown_cell("""### ROC"""))
    template = env.get_template('ROC.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    #Generate Provenance Data
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Generate Provenance Data"""))
    template = env.get_template('GenerateProvenanceData.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #Write Provenance Data
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Write Provenance Data"""))
    template = env.get_template('WriteProvenanceData.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))
    
    #Open Provenance Data Button
    nb['cells'].append(nbf.v4.new_markdown_cell("""### Show Provenance Data"""))
    template = env.get_template('OpenProvenanceData.jinja')
    output = template.render()
    nb['cells'].append(nbf.v4.new_code_cell(output))

    nbf.write(nb, 'GeneratedNotebooks/MulticlassClassification.ipynb')
    
    

    reply = {"greetings": "success"}
    return reply