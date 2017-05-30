var assert = require('assert');
var AdaBoost = require('../AdaBoost');

var trainingData = [
	["Coughing", "Male", "Adult", "Discharged"],
	["Coughing", "Female", "Teen", "Discharged"],
	["Headache", "Male", "Child", "Discharged"],
	["Headache", "Male", "Teen", "Discharged"],
	["Hiccups", "Female", "Adult", "Discharged"],
	["Sneezing", "Male", "Teen", "Discharged"],
	["Sneezing", "Female", "Child", "Admitted"],
	["Sneezing", "Male", "Child", "Admitted"],
	["Hiccups", "Female", "Teen", "Admitted"],
	["Coughing", "Female", "Adult", "Admitted"],
	["Coughing", "Male", "Child", "Admitted"]
];

var adaBoost = new AdaBoost(trainingData, "Discharged");

describe('AdaBoost', function () {
	describe('votes', function () {
		it('should return -1.7181763612259795', function () {
			var vote = adaBoost.predict(["Coughing", "Male", "Child"], true);
			assert.equal(-1.7181763612259795, vote);
		});

		it('should return 1.60653649762956', function () {
			var vote = adaBoost.predict(["Headache", "Female", "Child"]);
			assert.equal(1.60653649762956, vote);
		});

		it('should return -0.0843346278612036', function () {
			var vote = adaBoost.predict(["Hiccups", "Female", "Adult"]);
			assert.equal(-0.0843346278612036, vote);
		});

		it('should return 2.3552095483256945', function () {
			var vote = adaBoost.predict(["Headache", "", "Child"]);
			assert.equal(2.3552095483256945, vote);
		});
	});
});

