//
// Created by lucas on 18/04/19.
//

#ifndef NEURAL_NET_IN_CPP_LRSCHEDULER_H
#define NEURAL_NET_IN_CPP_LRSCHEDULER_H


class LRScheduler {
public:
    double learning_rate;
    virtual void onIterationEnd(int iteration) = 0;
};


#endif //NEURAL_NET_IN_CPP_LRSCHEDULER_H
