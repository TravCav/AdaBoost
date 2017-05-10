AdaBoost
========

Adaptive Boost Algorithm

#Example

(function () {
	'use strict';

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
		["Coughing", "Female", "Adult", "Admitted"]
	];

	var adaBoost = new AdaBoost(trainingData, "Discharged");

	console.log(adaBoost.trainedLearners);
	console.log("");

	var vote = adaBoost.predict(["Coughing", "Male", "Child"],true);
	console.log("The final vote is " + vote);
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
	console.log("");

	vote = adaBoost.predict(["Headache", "Female", "Child"]);
	console.log("The final vote is " + vote); 
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
	console.log("");

	vote = adaBoost.predict(["Hiccups", "Female", "Adult"]);
	console.log("The final vote is " + vote);
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
	console.log("");

	vote = adaBoost.predict(["Headache", "", "Child"]);
	console.log("The final vote is " + vote);
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
})();