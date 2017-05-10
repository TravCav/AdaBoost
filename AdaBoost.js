/**
 * @param {any} rawTrain is the dataset with the last column having the outcome you're looking for. 
 * @param {any} positiveOutcome is the value in the last column you want to predict for. 
 */
function AdaBoost(rawTrain, positiveOutcome) {
	'use strict';

	function Learner(data) {
		var self = this;
		self.desc = data.desc;
		self.feature = data.feature;
		self.predicted = data.predicted;
		self.epsilon = data.epsilon;
		self.alpha = -1;
		self.learned = false;
	}

	this.trainedLearners = TrainData(rawTrain, positiveOutcome);

	/**
	 * @param {any} testData A record without an outcome
	 * @returns positive number for true, negative number for false
	 */
	this.predict = function (testData) {
		// loop through each learner to see if it's needed.
		// if it is then accumulate the predicted * alpha
		var accumulatedVote = 0;
		for (var i = 0, learnerCount = this.trainedLearners.length; i < learnerCount; i++) {
			var learner = this.trainedLearners[i];
			if (learner.desc === testData[learner.feature]) {
				accumulatedVote += learner.predicted * learner.alpha;
			}
		}

		return accumulatedVote;
	};

	function TrainData(rawTrain, positiveOutcome) {
		var i; // for my loops. because javascript scoping and hoisting.

		// Generate a values matrix
		var values = [];

		var rows = rawTrain.length;
		var cols = rawTrain[0].length;
		for (var col = 0; col < cols; col++) {
			values[col] = [];
			var colLength = 0;
			for (var row = 0; row < rows; row++) {
				if (values[col].indexOf(rawTrain[row][col]) === -1) {
					values[col][colLength++] = rawTrain[row][col];
				}
			}
		}

		// Make Learners
		var dataRows = rawTrain.length;
		var dataCols = rawTrain[0].length;
		var learners = [];
		var learnerCount = 0;

		for (var featureIndex = 0, valueCount = values.length; featureIndex < valueCount - 1; featureIndex++) {
			for (var valueIndex = 0, featureCount = values[featureIndex].length; valueIndex < featureCount; valueIndex++) {
				var currentValue = values[featureIndex][valueIndex];

				// find how often each value occurs and what it's outcome was
				// return whether the value resulted more in a Discharged or Admitted
				// create a learner and add it to list or learners. Don't add if we couldn't figure out a Discharged or Admitted.
				var plusOne = 0,
					minusOne = 0,
					relevant = 0,
					predicted = 0;

				for (var dataRow = 0; dataRow < dataRows; dataRow++) {
					if (rawTrain[dataRow][featureIndex] === currentValue) {
						relevant++;
						(rawTrain[dataRow][dataCols - 1] !== positiveOutcome) ? minusOne++ : plusOne++;
					}
				}

				if (relevant !== 0 && plusOne !== minusOne) {
					// which one had more?
					predicted = (plusOne > minusOne) ? 1 : -1;
					var epsilon = 0;
					if (predicted === 1) {
						epsilon = minusOne * (1 / dataRows);
					} else {
						epsilon = plusOne * (1 / dataRows);
					}

					learners[learnerCount++] = new Learner({
						desc: values[featureIndex][valueIndex],
						feature: featureIndex,
						predicted: predicted,
						epsilon: epsilon,
					});
				}
			}
		}

		var trainWeights = [dataRows];
		var lastColumn = rawTrain[0].length - 1;

		// Initialize starting training weights
		var startWeight = 1.0 / dataRows;
		for (i = 0; i < dataRows; i++) {
			trainWeights[i] = startWeight;
		}

		// loop for however many learners there are.
		for (var learnIndex = 0; learnIndex < learnerCount; learnIndex++) {

			// update epsilons
			// loop through each learner and find it in the data
			// if the predicted value doesn't match then accumulate the training weight.
			// when all is said and done, assign the epsilon the accumulated training weight.
			// lather, rinse, repeat for the next learner.
			for (var updateIndex = 0; updateIndex < learnerCount; updateIndex++) {
				var learner = learners[updateIndex],
					ep = 0;
				for (i = 0; i < dataRows; i++) {
					// if this row has the same value as the learner and the prediction is wrong then accumulate the training weight.
					if (learner.desc === rawTrain[i][learner.feature] && learner.predicted !== (rawTrain[i][lastColumn] === positiveOutcome ? 1 : -1)) {
						ep += trainWeights[i];
					}
				}

				learners[updateIndex].epsilon = ep;
			}

			// find best learner. a.k.a. The unused one with the lowest epsilon.
			var bestLearner = -1;
			var lowestEpsilon = 99999999;
			for (var findIndex = 0; findIndex < learnerCount; findIndex++) {
				if (!learners[findIndex].learned && learners[findIndex].epsilon < lowestEpsilon) {
					lowestEpsilon = learners[findIndex].epsilon;
					bestLearner = findIndex;
				}
			}

			// assign to something not zero. otherwise we get divide by zero later.  Also, the smaller this number is, the bigger 0 becomes.  (there's probably a better way to explain that)
			if (lowestEpsilon === 0) {
				lowestEpsilon = 0.000001;
			}

			learners[bestLearner].learned = true;
			var alpha = 0.5 * Math.log((1.0 - lowestEpsilon) / lowestEpsilon); // increases greatly the further epsilon was from .5
			learners[bestLearner].alpha = alpha;

			var bLearner = learners[bestLearner];

			// update training weights by finding training data that matches the learner and scale it.
			for (i = 0; i < dataRows; i++) {
				if (bLearner.desc === rawTrain[i][bLearner.feature]) {
					trainWeights[i] = trainWeights[i] * Math.exp(-alpha * (rawTrain[i][lastColumn] === positiveOutcome ? 1 : -1) * bLearner.predicted);
				}
			}

			// total the training weights then divide each weight by total.
			var weightTotals = 0;
			for (i = 0; i < dataRows; i++) {
				weightTotals += trainWeights[i];
			}

			for (i = 0; i < dataRows; i++) {
				trainWeights[i] = trainWeights[i] / weightTotals;
			}

			// Do that all over again for the next best learner until they've all been used.
		}

		return learners;
	}
}

module.exports = AdaBoost;