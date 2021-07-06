<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\Persisters\Filesystem;

ini_set('memory_limit', '-1');

$logger = new Screen();

$estimator = new PersistentModel(
    new Pipeline([
        new NumericStringConverter(),
        new ZScaleStandardizer(),
    ], new MultilayerPerceptron([
        new Dense(128),
        new Activation(new ELU()),
        new Dense(64),
        new Activation(new ELU()),
        new Dense(32),
        new Activation(new ELU()),
        new Dense(16),
        new Activation(new ELU()),
    ], 256, new Momentum(0.01))),
    new Filesystem('model.rbx')
);

$estimator->setLogger($logger);

foreach (glob("datasets/train_*.csv") as $i => $file) {
    $logger->info("Loading $file into memory");

    $dataset = Labeled::fromIterator(new CSV($file));

    $logger->info('Partial training');

    $estimator->partial($dataset);

    $extractor = new CSV("progress_{$i}.csv", true);

    $extractor->export($estimator->steps());

    $logger->info("Progress saved to progress_{$i}.csv");
}

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator->save();

    $logger->info('Model saved to model.rbx');
}
