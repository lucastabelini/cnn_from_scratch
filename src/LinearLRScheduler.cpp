//
// Created by lucas on 18/04/19.
//

#include "../include/LinearLRScheduler.h"


LinearLRScheduler::LinearLRScheduler(double initial_lr, double step) {
    learning_rate = initial_lr;
    this->step = step;
}

void LinearLRScheduler::onIterationEnd(int iteration) {
    learning_rate += step;
}
