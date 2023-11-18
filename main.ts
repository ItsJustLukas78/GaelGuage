import "@tensorflow/tfjs-node"
import * as tf from '@tensorflow/tfjs';
import fs from 'fs';

const model: tf.Sequential = tf.sequential();

model.add(tf.layers.dense({ inputShape: [8], units: 64, activation: 'relu' }));
model.add(tf.layers.dropout({ rate: 0.5 }));
model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
model.add(tf.layers.dropout({ rate: 0.5 }));
model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

model.compile({
  loss: 'binaryCrossentropy',
  optimizer: 'adam',
  metrics: ['accuracy']
});


// Training data
const csvDataset = tf.data.csv('https://docs.google.com/spreadsheets/d/18X2QZf-XRKa3ZtjP4oSYbu5ZiLeXAVBZ3MyJSGKVT2s/gviz/tq?tqx=out:csv', {
  columnConfigs: {
    red_won: {
      isLabel: true
    }
  }
});


const numOfSamples = 706;

const convertedData = csvDataset.map(({xs, ys}: any) => {
  const labels = [
    ys.red_won
  ];

  return {xs: Object.values(xs), ys: Object.values(labels)};
}).batch(25);


// Testing data
const csvTestDataset = tf.data.csv('https://docs.google.com/spreadsheets/d/1oGkVQQ-vf_JDG7STtZ5DIgok0T5ZsVBPQBQClOfhbQ4/gviz/tq?tqx=out:csv', {
  columnConfigs: {
    red_won: {
      isLabel: true
    }
  }
});

const numOfTestSamples = 260;

const convertedTestData = csvTestDataset.map(({xs, ys}: any) => {
  const labels = [
    ys.red_won
  ];

  return {xs: Object.values(xs), ys: Object.values(labels)};
}).batch(260);


model.fitDataset(convertedData, {
  epochs: 450,
  validationData: convertedTestData,
  callbacks: {
    onEpochEnd: async (epoch:any, logs:any) => {
      console.log(`Epoch: ${epoch}, Loss: ${logs.loss} Accuracy: ${logs.acc} Validation loss: ${logs.val_loss} Validation accuracy: ${logs.val_acc}`);
    },
}}).then(() => {
  console.log('Training complete');
  model.save('file://./predictionModel').then(() => {
    console.log('Model saved');
  });

  // Use test data to evaluate model
  model.evaluateDataset(convertedTestData as any, {}).then((result: any) => {
    console.log(`Test loss: ${result[0]} Test accuracy: ${result[1]} `);
  });

  // extract from test.csv and loop through each row

  fs.readFile('./test.csv', 'utf8', (err, data) => {
    if (err) {
      console.error(err);
      return;
    }
    const lines = data.split(/\r?\n/);
    for (const line of lines) {
      // skip first line
      if (line.startsWith('red_opr')) {
        continue;
      }
      const values = line.split(',');
      if (values.length === 9) {
        const red_opr = parseFloat(values[0]);
        const red_dpr = parseFloat(values[1]);
        const red_ccwm = parseFloat(values[2]);
        const red_openSkill = parseFloat(values[3]);
        const blue_opr = parseFloat(values[4]);
        const blue_dpr = parseFloat(values[5]);
        const blue_ccwm = parseFloat(values[6]);
        const blue_openSkill = parseFloat(values[7]);
        const red_won = parseInt(values[8]);
        const resultTensor: any = model.predict(tf.tensor2d([red_opr, red_dpr, red_ccwm, red_openSkill, blue_opr, blue_dpr, blue_ccwm, blue_openSkill], [1, 8]));
        const predictedValue = resultTensor.arraySync()[0][0];
        console.log(`Predicted: ${predictedValue}, Actual: ${red_won}`);
      }
    }
  });

});


// const model = await tf.loadLayersModel('file://./predictionModel/model.json');
//
// const resultTensor: any = model.predict(tf.tensor2d([130.8654,67.6436,63.2218,25.1906,69.5607,93.7254,-24.1647,-1.1934], [1, 8]));
// const predictedValue = resultTensor.arraySync()[0][0];
//
// console.log(predictedValue);
