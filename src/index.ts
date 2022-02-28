import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, MainAreaWidget } from '@jupyterlab/apputils';
import { Widget } from '@lumino/widgets';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { requestAPI } from './handler';

/* eslint-disable no-useless-escape */


async function activate (app: JupyterFrontEnd, palette: ICommandPalette, settingRegistry: ISettingRegistry | null) {
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
  // Add an application command
  const command = 'codegenerator:open';
  app.commands.addCommand(command, {
    label: 'Code Generation from Provenance data',
    execute: () => {
      if (!widget.isAttached) {
        // Attach content to the main work area if it's not there
        app.shell.add(widget, 'main');
      }
      // Activate the widget
      app.shell.activateById(widget.id);
    }
  });
  // ------------------------------------------------------------------------------------------------------------------------------- //
  // Add the command to the palette.
  palette.addItem({ command, category: 'Tutorial' });	
  } 
/**
 * Initialization data for the extension extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'extension:plugin',
  autoStart: true,
  requires: [ICommandPalette],
  optional: [ISettingRegistry],
  activate: activate
}
    

export default plugin;
