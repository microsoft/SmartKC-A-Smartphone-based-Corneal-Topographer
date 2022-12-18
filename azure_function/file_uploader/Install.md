# Steps to set up backend
-------------------------------
Below instruction is to setup backend to automatically upload the generated files in the android app to the Blob store.
## Setting up the azure function
1. Search for Function App in the market place
2. Click on create a function app, in the form that appears:
  0. Modify the prefilled fields if required. 
  1. Fill in the azure function app name. Example: smartKcFunctions
  2. Choose runtime to be `Python`
  3. Choose operating system as `Linux`
3. Click on `Next > Hosting`
4. Select a storage account or else create a new storage account if there is no existing storage account
5. Click on `Review + Create` from top
## Setup local environment for Azure function
Once, you have created a function app, you need to create a local environment to publish a new function under the created function app.
### Prerequisites
1. [The Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=v4%2Clinux%2Ccsharp%2Cportal%2Cbash#install-the-azure-functions-core-tools) version 4.x.
2. Python versions that are supported by Azure Functions. For more information, see How to install [Python](https://wiki.python.org/moin/BeginnersGuide/Download).
3. [Visual Studio Code](https://code.visualstudio.com/)
4. The [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) for Visual Studio Code.
5. The [Azure Functions extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-azurefunctions) for Visual Studio Code, version 1.8.3 or a later version.

### Create a local project
1. Open a new empty folder with Visual Studio Code
2. Choose the Azure icon in the Activity bar. Then in the `Workspace (local) area`, select the `+` button
3. Choose `Create HTTP function`
4. Select `Python` when prompted for select a language.
5. Select `Skip virtual environment` when prompted for selecting interpreter path.
6. Name your Http function. For eg. `fileUploader`.
7. From the github repository, Under the folder `azure_function/uploader/smart_kc_test_uploader` copy files `function.json` and `host.json` and paste it to your respective `function.json` and `host.json`
8. Copy the above step with `azure_function/uploader/requirements.txt` and paste it to your respective `requirements.txt`
9. Make sure the app runs by typing: `func start` in the terminal open inside `azure_function/uploader`
### Publish the function
1. Publish the function by running the command `func azure functionapp publish [YourFunctionAppName]`

### Set Environment for the function
Certain variables need to be setup for the azure function. Follow these steps to set this up:
1. Go to the Azure Function resource
2. Click on the `Configuration` under `Settings` section
3. Under `Application settings` tab click on `New application setting`
4. On the new dialogue box that appears, enter name of the variable as `upload_secret` and put in a strong secret. 
5. Click `OK` to save the variable.
6. Click `Save` at the top to save the application settings.
7. Click `Continue` on the warning if it appears.

### Give Blob permissions to Azure function
To upload files in the blob storage azure function needs certain permissions. Follow below steps to set it up.
1. Go to your Azure function App.
2. Click on `Identity` under `Settings`
3. Under the `System Assigned` tab, turn the `Status` switch to On and click `Save` on the top.
4. After you click `Save`, Click the button that will appear as `Add Azure Role Assignments`.
5. On the new page that appears, Click `Add role Assignments`.
6. Next, select scope as `Storage`. Then, select a storage resource where you want the files to be uploaded. Further, select the role as `Blob Storage Contributor role`
7. Click `Save`

### Get URL for Azure function
To get the URL of Azure function, go to the function app in your azure portal. Under `Funcions` click the Azure function created by you and click on `Get Function Url` present on the top.
