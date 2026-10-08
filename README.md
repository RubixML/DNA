# DNA Taxonomer

An example project demonstrating the use of machine learning to identify microbe taxonomies from their DNA sequences.

## Requirements

- [PHP](https://php.net) 8.3 or above.
- [Tensor 4.1+ extension](https://github.com/RubixML/Tensor-Ext) for fast training and inference.

## Installation

Clone the project locally using [Composer](https://getcomposer.org) or install it as a package:

```sh
composer create-project rubix/dna
```

> **Note:** Installation may take longer than usual because of the large dataset.

Then install the [Tensor Ext](https://packagist.org/packages/rubix/tensor_ext) extension using [PIE](https://github.com/php/pie) like in the example below.

```sh
pie install rubix/tensor_ext:^4.1
```

## Tutorial

### Introduction

Our objective is to predict the biological taxonomy of a DNA sequence using machine learning. More precisely, given a short piece of DNA we want to classify it into one of five broad groups: `virus`, `bacteria`, `animal`, `fungi`, or `plant`. This is a common task in [metagenomics](https://en.wikipedia.org/wiki/Metagenomics) where we need to identify which organisms are present in an environmental sample.

We'll represent each DNA sequence as a [k-mer count](https://en.wikipedia.org/wiki/Crossmatch#K-mer) feature vector. A k-mer is a substring of length k drawn from the sequence, and counting how often each of them appears produces a fixed-length feature vector independent of how long the sequence we started with happened to be. We'll use k equal to 4, which gives us 4^4 = 256 possible k-mer features. The full list of features in the order we will use them can be found in the `features.json` file.

> For the DNA sequence `GCAATG` the 4-mers we find are `GCAT`, `CAAT`, and `AATG`. If another sample happened to contain `GCAT` twice, `CAAT` once, and `AATG` once we would set the count of those features to 2, 1, and 1 - and the remaining 253 features in the vector to 0.

The dataset provided to us contains 4 training files (`datasets/train_1.csv` - `datasets/train_4.csv`) and one test file (`datasets/test.csv`). Each file contains roughly 35,000 samples, and combined we have close to 140,000 training samples in total. We'll use the four training files to train the model and the test file to validate the model's performance. From there, we'll use the dataset to train a multilayer neural network - the [Multilayer Perceptron](https://rubixml.github.io/ML/latest/classifiers/multilayer-perceptron.html) - to classify the taxonomy of any DNA sequence we show it.

### Extracting the Data

Our samples are given to us as [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) files. Each row contains a single sample - the 256 k-mer counts as the first 256 columns and the label in the last column. The four training files are combined into a single stream with the [Concatenator](https://rubixml.github.io/ML/latest/extractors/concatenator.html) extractor and each file is loaded individually with the [CSV](https://rubixml.github.io/ML/latest/extractors/csv.html) extractor.

```php
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\Concatenator;
use Rubix\ML\Datasets\Labeled;

define('CHUNK_SIZE', 61440);

$extractor = new Concatenator([
    new CSV('datasets/train_1.csv'),
    new CSV('datasets/train_2.csv'),
    new CSV('datasets/train_3.csv'),
    new CSV('datasets/train_4.csv'),
]);
```

> **Note**: The source code for this example can be found in the [train.php](https://github.com/RubixML/DNA/blob/master/train.php) file in the project root.

Unlike the [Sentiment](https://github.com/RubixML/Sentiment) example, where we loaded the entire dataset into memory, we are going to iterate over the data in chunks. This is because the dataset is large and may not fit into memory on all machines. We use the static `Labeled::chunked()` method to load the data in blocks of 61,440 samples, defined by the `CHUNK_SIZE` constant. That number is a multiple of the model's batch size, so every full block yields a whole number of gradient batches. When the final block is not full the `Labeled` iterator will yield a partial block - the size of an iteration of a chunked extractor is not guaranteed to be constant.

> **Note**: The data is loaded lazily. No samples are read until we actually iterate over the extractor.

### Dataset Preparation

Neural nets compute a non-linear continuous function and therefore require continuous features as inputs. However, CSV data is imported as categorical (string) data by default. We'll convert the k-mer counts to continuous values using the [Float Type Converter](https://rubixml.github.io/ML/latest/transformers/float-type-converter.html) transformer.

After conversion we normalize the features to a range between 0 and 1 - which reduces the variance of the input features to a single order of magnitude. We do so using the [Z Scale Standardizer](https://rubixml.github.io/ML/latest/transformers/z-scale-standardizer.html) to center and scale the feature matrix to have 0 mean and unit variance. This last step will help the neural network converge quicker.

```php
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformer = new PersistentTransformer(
    base: new Pipeline([
        new FloatTypeConverter(),
        new ZScaleStandardizer(),
    ]),
    persister: new Filesystem('transformer.rbx')
);
```

This is a much simpler preparation pipeline than the text example because our features are already numeric. All that's left is to ensure they are of the right data type, and centered and scaled.

Note that the pipeline is wrapped in a [Persistent Transformer](https://rubixml.github.io/ML/latest/transformers/persistent-transformer.html) rather than being part of the model. This lets us save the transformer to `transformer.rbx` and reload it in the validation script so that the test set is transformed with the exact same pipeline used during training.

### Instantiating the Learner

Now we'll define the architecture of the neural network and instantiate the [Multilayer Perceptron](https://rubixml.github.io/ML/latest/classifiers/multilayer-perceptron.html) classifier. The network uses 6 hidden blocks. Each block consists of a [Dense](https://rubixml.github.io/ML/latest/neural-network/hidden-layers/dense.html) layer of 256 neurons followed by a non-linear [Activation](https://rubixml.github.io/ML/latest/neural-network/hidden-layers/activation.html) layer that applies the [SiLU](https://rubixml.github.io/ML/latest/neural-network/activation-functions/silu.html) (swish) activation function. Every other block also includes a [Batch Norm](https://rubixml.github.io/ML/latest/neural-network/hidden-layers/batch-norm.html) layer to normalize the activations of the neurons before they are passed to the activation function. On those blocks the `Dense` layer is created with `bias: false` since the batch norm layer introduces a bias term of its own - doubling up would be redundant.

The activation function is what gives the network the ability to learn non-linear relationships between the input and output features. We've found that using a mixture of `Dense` and `BatchNorm` layers with SiLU activations works fairly well for this class of problem. The number and size of each layer is a hyper-parameter you can tune.

```php
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;

$mlp = new MultilayerPerceptron(
    hiddenLayers: [
        new Dense(256),
        new Activation(new SiLU()),
        new Dense(256, bias: false),
        new BatchNorm(),
        new Activation(new SiLU()),
        new Dense(256),
        new Activation(new SiLU()),
        new Dense(256, bias: false),
        new BatchNorm(),
        new Activation(new SiLU()),
        new Dense(256),
        new Activation(new SiLU()),
        new Dense(256, bias: false),
        new BatchNorm(),
        new Activation(new SiLU()),
        new Dense(5)
    ]
);
```

The output layer has the same number of neurons as there are classes in the dataset - 5 in this case.

We'll choose a *batch size* of 32 samples per gradient update. We also set `gradientAccumulationSteps` to 4 which accumulates gradients over 4 batches before performing an update - effectively increasing the batch size by a factor of 4 while keeping memory usage the same.

```php
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$mlp = new MultilayerPerceptron(
    hiddenLayers: [...],
    batchSize: 32,
    gradientAccumulationSteps: 4,
    optimizer: new AdaMax(new Constant(0.0001)),
    epochs: 100,
    minChange: 1e-5,
    evalInterval: 1,
    window: 10
);
```

We use the [AdaMax](https://rubixml.github.io/ML/latest/neural-network/optimizers/adamax.html) optimizer with a *learning rate* of 0.0001. When setting the learning rate of an optimizer the important thing to note is that a rate that is too low will cause the network to learn slowly while a rate that is too high will prevent the network from learning at all.

We'll wrap the model in a [Persistent Model](https://rubixml.github.io/ML/latest/persistent-model.html) wrapper so we can save and load it later in our other scripts. The [Filesystem](https://rubixml.github.io/ML/latest/persisters/filesystem.html) persister tells the wrapper to save and load the serialized model data from `model.rbx` on disk.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(
    base: $mlp,
    persister: new Filesystem('model.rbx')
);
```

Note that the transformer is *not* included in the estimator - it is applied to the data ahead of time (as you'll see in the next section) and persisted to its own file so that we can reuse the exact same pipeline at inference time.

Last, we'll create a [Screen](https://rubixml.github.io/ML/latest/loggers/screen.html) logger and attach it to the estimator so that we can see training progress in the console. We reuse the same logger instance for the other status messages logged throughout the script.

```php
use Rubix\ML\Loggers\Screen;

$logger = new Screen();

$estimator->setLogger($logger);
```

### Training

Before training begins, we load the testing set and set it as the model's *validation* dataset using the `setValidationDataset()` method. This fixes the validation set for the whole run so that the early-stopping metric is computed on the same, held-out samples at every epoch, rather than on a slice of each incoming chunk. We also apply the transformer to it so that the model's internals - which expect continuous, normalized inputs - see the correct representation.

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;

$testing = Labeled::fromIterator(new CSV('datasets/test.csv'));

$testing->apply($transformer);

$estimator->setValidationDataset($testing);
```

Now you can call the `partial()` method on the learner with a dataset as an argument to kick off the training process. `partial()` allows the learner to incrementally learn from new data. This is useful when we can't fit the whole dataset into memory because we can feed it in blocks - each block is used to update the model's parameters before moving on to the next. We wrap the chunked extractor with `enumerate()` to keep track of which block we're on.

Because the transformer is updated as it sees new data (the online-style `Z Scale Standardizer` accumulates running statistics), we refresh it with each chunk via `$transformer->update($dataset)` and then transform the chunk before it is fed to the model with `$dataset->apply($transformer)`:

```php
use Rubix\ML\Datasets\Labeled;

use function Rubix\ML\enumerate;

foreach (enumerate(Labeled::chunked($extractor, size: CHUNK_SIZE), start: 1) as $i => $dataset) {
    $transformer->update($dataset);
    $dataset->apply($transformer);
    $estimator->partial($dataset);
}
```

During training, the learner will record the validation score and the training loss at each iteration or *epoch*. The validation score is calculated using the default [F Beta](https://rubixml.github.io/ML/latest/cross-validation/metrics/f-beta.html) metric on the fixed *validation* set we set above. Contrariwise, the training loss is the value of the cost function (in this case the [Cross Entropy](https://rubixml.github.io/ML/latest/neural-network/cost-functions/cross-entropy.html) loss) calculated over the incoming block of samples. We can visualize the training progress by plotting these metrics. To output the scores and losses you can call the additional `progress()` method and pass the resulting iterator to a Writable extractor such as [CSV](https://rubixml.github.io/ML/latest/extractors/csv.html).

Since `partial()` resets the progress table at the start of every block, we export it to its own file after each one - named after the block number so that nothing is clobbered. The second argument to the CSV extractor constructor marks the file as writable, and `overwrite: true` lets us replace a file left over from a previous run.

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV("progress_{$i}.csv", true);

$extractor->export($estimator->progress(), overwrite: true);
```

> **Note:** When training a network incrementally with `partial()` - rather than `train()` - the hyper-parameters controlling early-stopping are `evalInterval` (how often to evaluate - once per epoch in our case) and `window` (how many epochs without improvement before early-stopping). The validation set used to measure improvement is the one we fixed above with `setValidationDataset()`.

The validation score should be getting better with each epoch as the loss decreases. You can generate your own plots by importing the `progress_1.csv`, `progress_2.csv`, etc. files into your plotting application. Because each file covers a different block of the dataset, plot them in order to follow the model across the whole training run.

Finally, we save both the transformer and the model so we can load them later in our validation script.

```php
if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $transformer->save();
    $estimator->save();
}
```

Now you're ready to run the training script from the command line.

```sh
php train.php
```

### Cross Validation

To test the generalization performance of the trained network we'll use the testing samples provided to us to generate predictions and then analyze them compared to their ground-truth labels using a cross-validation report. Note that we do not use any training data for cross validation because we want to test the model on samples it has never seen before.

> **Note**: The source code for this example can be found in the [validate.php](https://github.com/RubixML/DNA/blob/master/validate.php) file in the project root.

We'll start by importing the testing samples from `datasets/test.csv`. The samples and labels are loaded into a [Labeled](https://rubixml.github.io/ML/latest/datasets/labeled.html) dataset object using the static `fromIterator()` method.

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;

$dataset = Labeled::fromIterator(new CSV('datasets/test.csv', true));
```

Next, we'll use the Persistent Transformer and Persistent Model wrappers to load the transformer and network we saved during training. The transformer must be applied to the testing samples before they are fed to the model - the model was trained on normalized inputs and will not behave correctly on raw data. Calling `cleanup()` releases any resources held by the loaded model - it is purely an optimization and can be safely omitted.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Persisters\Filesystem;

$transformer = PersistentTransformer::load(new Filesystem('transformer.rbx'));

$estimator = PersistentModel::load(new Filesystem('model.rbx'));
$estimator->cleanup();

$dataset->apply($transformer);
```

Now we can use the estimator to make predictions on the testing set. The `predict()` method on the estimator takes a dataset as input and returns an array of predictions.

```php
$predictions = $estimator->predict($dataset);
```

The cross-validation report we'll generate is actually a combination of two reports - [Multiclass Breakdown](https://rubixml.github.io/ML/cross-validation/reports/multiclass-breakdown.html) and [Confusion Matrix](https://rubixml.github.io/ML/cross-validation/reports/confusion-matrix.html). We wrap each report in an [Aggregate Report](https://rubixml.github.io/ML/cross-validation/reports/aggregate-report.html) to generate both reports at once. The Multiclass Breakdown will give us detailed information about the performance of the estimator at the class level. The Confusion Matrix will give us an idea as to what labels the estimator is confusing one another for by binning the labels in a 5 x 5 matrix.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);
```

To generate the report, pass in the predictions along with the labels from the testing set to the `generate()` method on the report. The return value is a report object that can be echoed out to the console or saved to a file in JSON form.

```php
$results = $report->generate($predictions, $dataset->labels());

$results->toJSON()->saveTo(new Filesystem('report.json'));
```

Now we can execute the validation script from the command line.

```sh
php validate.php
```

Take a look at the report and see how well the model performs. A well-trained model should have an accuracy well above the 20% that we would achieve by guessing randomly (since there are 5 classes). If your model's accuracy is near or below 20% then the model is probably overfit or was not trained for enough epochs.

### Next Steps

Congratulations on completing the tutorial on DNA taxonomy classification in Rubix ML using a Multilayer Perceptron. We recommend playing around with the network architecture and hyper-parameters on your own to get a feel for how they affect the model. Generally, adding more neurons and layers will improve performance but training may take longer as a result. You could also experiment with different [activation functions](https://rubixml.github.io/ML/latest/neural-network/activation-functions.html), [optimizers](https://rubixml.github.io/ML/latest/neural-network/optimizers.html), or feature engineering approaches (for example a larger k-mer size) and see which combination works best for this problem.

## References

>- M. H. Mohammed et al. (2011). Eu-Detect: An algorithm for detecting eukaryotic sequences in metagenomic data sets.

## License

The code is licensed [MIT](LICENSE) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
