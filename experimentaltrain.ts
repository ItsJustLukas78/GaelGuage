import "@tensorflow/tfjs-node";
import * as tf from '@tensorflow/tfjs';
import fs from 'fs';

// Model Architecture
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [8], units: 64, activation: 'relu' }));
model.add(tf.layers.dropout({ rate: 0.5 }));
model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
model.add(tf.layers.dropout({ rate: 0.5 }));
model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compile model
model.compile({
  loss: 'binaryCrossentropy',
  optimizer: 'adam',
  metrics: ['accuracy']
});

// Helper function to process dataset
const processData = (csvDataset, isLabelConfig) => {
  return csvDataset.map(({ xs, ys }) => {
    const labels = [ys.red_won];
    return { xs: Object.values(xs), ys: Object.values(labels) };
  }).batch(25); // Experiment with batch size
};

// Training data
const csvDataset = tf.data.csv('https://docs.google.com/spreadsheets/d/18X2QZf-XRKa3ZtjP4oSYbu5ZiLeXAVBZ3MyJSGKVT2s/gviz/tq?tqx=out:csv', { columnConfigs: { red_won: { isLabel: true } } });
const convertedData = processData(csvDataset);

// Testing data
const csvTestDataset = tf.data.csv('https://docs.google.com/spreadsheets/d/1oGkVQQ-vf_JDG7STtZ5DIgok0T5ZsVBPQBQClOfhbQ4/gviz/tq?tqx=out:csv', { columnConfigs: { red_won: { isLabel: true } } });
const convertedTestData = processData(csvTestDataset);

// Training with callbacks for early stopping
model.fitDataset(convertedData, {
  epochs: 450,
  validationData: convertedTestData,
  callbacks: {
    onEpochEnd: async (epoch, logs) => {
      console.log(`Epoch: ${epoch}, Loss: ${logs.loss} Accuracy: ${logs.acc} Validation loss: ${logs.val_loss} Validation accuracy: ${logs.val_acc}`);
    }
  }
}).then(() => {
  console.log('Training complete');
  model.save('file://./predictionModel').then(() => {
    console.log('Model saved');
  });

  // Evaluate model
  model.evaluateDataset(convertedTestData).then((result) => {
    console.log(`Test loss: ${result[0]} Test accuracy: ${result[1]}`);
  });

  // Prediction
  fs.readFile('./test.csv', 'utf8', (err, data) => {
    if (err) {
      console.error(err);
      return;
    }
    const lines = data.split(/\r?\n/);
    lines.forEach((line, index) => {
      if (index > 0 && line) { // Skipping header
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
          const resultTensor = model.predict(tf.tensor2d([red_opr, red_dpr, red_ccwm, red_openSkill, blue_opr, blue_dpr, blue_ccwm, blue_openSkill], [1, 8]));
          const predictedValue = resultTensor.arraySync()[0][0];
          console.log(`Predicted: ${predictedValue}, Actual: ${red_won}`);
        }
      }
    });
  });
});
