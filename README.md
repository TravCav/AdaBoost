AdaBoost
========

Adaptive Boost Algorithm

#Example

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

var Predict = AdaBoost.MakePredictor(trainingData, "Discharged");

var vote = Predict(["Coughing", "Male", "Child"]);
console.log("The final vote is " + vote);
console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));

vote = Predict(["Headache", "Female", "Child"]);
console.log("The final vote is " + vote);
console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));

vote = Predict(["Hiccups", "Female", "Adult"]);
console.log("The final vote is " + vote);
console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));

vote = Predict(["Headache", "", "Child"]);
console.log("The final vote is " + vote);
console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
