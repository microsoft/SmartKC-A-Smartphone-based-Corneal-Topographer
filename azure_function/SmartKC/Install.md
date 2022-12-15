# Steps to set up backend
-------------------------------
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
7. From the github repository
