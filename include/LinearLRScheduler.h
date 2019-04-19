//
// Created by lucas on 18/04/19.
//

#ifndef NEURAL_NET_IN_CPP_LINEARLRSCHEDULER_H
#define NEURAL_NET_IN_CPP_LINEARLRSCHEDULER_H

#include "LRScheduler.h"

class LinearLRScheduler : public LRScheduler {
public:
    double step;
    LinearLRScheduler(double initial_lr, double step);
    void onIterationEnd(int iteration) override;
};


#endif //NEURAL_NET_IN_CPP_LINEARLRSCHEDULER_H
