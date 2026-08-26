#include "DKP_PCLayer.h"

namespace Deep
{
    DKP_PCLayer::DKP_PCLayer(DKP_PCLayer *prev, DKP_PCLayer *next, size_t inSize, size_t outSize, float lr)
    {
        this->inSize = inSize;
        this->outSize = outSize;
        this->prevLayer = prev;
        this->nextLayer = next;
        this->prevLayer->nextLayer = this;
        this->nextLayer->prevLayer = this;
        this->lr = lr;

        feedback = std::make_unique<float>(outSize * nextLayer->outSize);
    }
};