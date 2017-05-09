AdaBoost
========

Adaptive Boost Algorithm

#Example

(function () {
	'use strict';

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