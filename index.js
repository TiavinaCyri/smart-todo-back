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
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

model.fit(X_tensor, y_tensor, { epochs: 140 }).then(() => {
    function predictPriorities(tasks) {
        const predictedPriorities = [];
        tasks.forEach(task => {
            const task_length = task.length;
            if (task_length === 0) {
                console.error("La longueur de la tâche ne peut pas être zéro.");
                predictedPriorities.push(null);
            } else {
                const task_length_normalized = task_length / X_max;
                const predicted_priority_normalized = model.predict(tf.tensor2d([[task_length_normalized]], [1, 1])).dataSync()[0];
                const predicted_priority = predicted_priority_normalized * y_max;
                predictedPriorities.push(predicted_priority);
            }
        });
        return predictedPriorities;
    }

    const tasks = ["Dormir", "Soigner les malades en urgence", "Répondre aux e-mails"];

    const predictedPriorities = predictPriorities(tasks);
    console.log("Priorités prédites des tâches :", predictedPriorities);
});