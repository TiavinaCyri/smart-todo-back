const tf = require('@tensorflow/tfjs-node');
const trainingData = require('./dataset.json');

const X = trainingData.map(d => d.task.length);
const y = trainingData.map(d => d.priority);

const X_max = Math.max(...X);
const y_max = Math.max(...y);

const X_normalized = X.map(val => val / X_max);
const y_normalized = y.map(val => val / y_max);

const X_tensor = tf.tensor2d(X_normalized, [X_normalized.length, 1]);
const y_tensor = tf.tensor2d(y_normalized, [y_normalized.length, 1]);

const model = tf.sequential();
model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [1] }));
model.add(tf.layers.dropout({rate: 0.5}));
model.add(tf.layers.dense({ units: 1 }));
// Compilaton du modèle
model.compile({
    optimizer: tf.train.adam(),
    loss: 'meanSquaredError',
    metrics: ['accuracy']
});

// Entraînement du modèle
async function trainModel(model, X_tensor, y_tensor){
    await model.fit(X_tensor, y_tensor, {
        epochs: 140,
        validationSplit: 0.2,
        callbacks: tf.callbacks.earlyStopping({ patience: 10 })
    });
}

function predictPriorities(tasks) {
    const taskLengths = tasks.map(task => task.length / X_max);

    const predictions = model.predict(tf.tensor2d(taskLengths, [taskLengths.length, 1])).dataSync();
    return predictions.map(pred => pred * y_max);
}

async function run(){
    await trainModel(model, X_tensor, y_tensor);

    const tasks = ["Dormir", "Soigner les malades", "Regarder le ciel"];
    const predictedPriorities = predictPriorities(tasks);
    
    console.log("Priorités prédites des tâches :", predictedPriorities);
}

run();