#include "PCNLayer.h"
#include <iostream>

int main(void)
{
    std::cout << "Started running batches test.\n";
    Deep::PCLayer layer(2, 8);
    float a[2] = {1,1};
    float b[2] = {2,2};
    float c[2] = {3,3};
    float d[2] = {4,4};

    layer.RunPrediction(a);
    layer.RunPrediction(b);
    layer.RunPrediction(c);
    layer.RunPrediction(d);

    float e[2] = {9,9};
    layer.RunPrediction(e);
    layer.Flush();

    // Should print 99 22 33 44
}