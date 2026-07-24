// Example Neuva programs loaded into the playground editor via the dropdown.
// Kept in sync with the language surface supported by neuva_web.py.

export const EXAMPLES = {
  "Hello World": `# Hello World in Neuva
let name = "Neuva"
print "Hello from", name

let version = 1.1
print "Playground running Neuva v{version}"
`,

  "Iris Classifier": `# Classic multi-class classification — Iris species from 4 flower measurements
let data = load("iris.csv")
let data = data.normalize()
let train_data, test_data = data.split(0.8)

model IrisClassifier {
    layer dense(4 -> 16, relu)
    layer dense(16 -> 8, relu)
    layer dense(8 -> 3, softmax)
}

print IrisClassifier

train IrisClassifier on train_data for 20 epochs, lr = 0.01, loss = crossentropy

let acc = accuracy(IrisClassifier, test_data)
print "Test accuracy: {acc}"
`,

  "Control Flow": `# if/else, for, while, and functions
fn square(x: int) -> int {
    return x * x
}

fn classify(n: int) -> string {
    if n < 0 {
        return "negative"
    } else if n == 0 {
        return "zero"
    } else if n < 10 {
        return "small"
    } else {
        return "large"
    }
}

for i in range(5) {
    let sq = square(i)
    print "square(", i, ") =", sq, "->", classify(sq)
}

let total = 0
let n = 1
while n <= 5 {
    let total = total + n
    let n = n + 1
}
print "sum 1..5 =", total

let nums = [3, 7, 2, 9, 4]
let biggest = nums[0]
for x in nums {
    if x > biggest {
        let biggest = x
    }
}
print "numbers:", nums
print "largest value seen:", biggest
`,

  "Custom Loss": `# Define a loss function in Neuva itself and use it during training
model Regressor {
    layer dense(8 -> 32, relu)
    layer dense(32 -> 16, relu)
    layer dense(16 -> 1, linear)
}

fn my_loss(pred, target) {
    # Penalize errors a bit more than plain MSE
    return mse(pred, target) * 1.2
}

let data = load("housing.csv")
let data = data.normalize()
let train_data, test_data = data.split(0.8)

train Regressor on train_data for 15 epochs, lr = 0.001, loss = my_loss

predict Regressor on test_data
print Regressor
`,
};
