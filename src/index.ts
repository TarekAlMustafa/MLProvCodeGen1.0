import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { ISettingRegistry } from '@jupyterlab/settingregistry';

import { requestAPI } from './handler';


async function activate (app: JupyterFrontEnd, settingRegistry: ISettingRegistry | null) {
	console.log('JupyterLab extension extension is activated!');

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
	  
	const dataToSend = { name: 'MLProvCodeGen' };
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
  } 
/**
 * Initialization data for the extension extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'extension:plugin',
  autoStart: true,
  optional: [ISettingRegistry],
  activate: activate
}
    

export default plugin;
