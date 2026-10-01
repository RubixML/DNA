<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\Concatenator;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\Persisters\Filesystem;

use function Rubix\ML\enumerate;

ini_set('memory_limit', '-1');

define('CHUNK_SIZE', 61440);

$logger = new Screen();

$extractor = new Concatenator([
    new CSV('datasets/train_1.csv'),
    new CSV('datasets/train_2.csv'),
    new CSV('datasets/train_3.csv'),
    new CSV('datasets/train_4.csv'),
]);

$estimator = new PersistentModel(
    new Pipeline([
        new FloatTypeConverter(),
        new ZScaleStandardizer(),
    ], new MultilayerPerceptron(
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
        ],
        batchSize: 32,
        gradientAccumulationSteps: 4,
        optimizer: new AdaMax(new Constant(0.0001)),
        epochs: 100,
        minChange: 1e-5,
        evalInterval: 1,
        window: 10
    )),
    new Filesystem('model.rbx')
);

$estimator->setLogger($logger);

foreach (enumerate(Labeled::chunked($extractor, size: CHUNK_SIZE), start: 1) as $i => $dataset) {
    $logger->info("Training chunk #{$i}");

    $estimator->partial($dataset);

    $extractor = new CSV("progress_{$i}.csv", true);

    $extractor->export($estimator->progress(), overwrite: true);

    $logger->info("Progress saved to progress_{$i}.csv");
}

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator->save();

    $logger->info('Model saved to model.rbx');
}
