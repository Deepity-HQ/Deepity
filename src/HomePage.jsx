import { Link } from "react-router-dom";
import Arrow from "./components/arrow";
import PythonIcon from "./assets/python.png";
import CppIcon from "./assets/cpp.png";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { docco } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { lazy, Suspense, useState, memo } from "react";
import MainFooter from "./components/MainFooter";
import CodeBlock from "./components/CodeBlock";

const BenchmarkChart = lazy(() => import("./components/BenchmarkChart"));
const PCNRepresentation = lazy(() => import("./components/PCNRepresentation"));

const pythoncode = `from pydeepity import SimplePCN
from pydeepity.layer import Linear, Sigmoid
import numpy as np

# Architecture is declared, not assembled layer-by-layer.
net = SimplePCN(
    Linear(784, 512), Sigmoid(),
    Linear(512, 512), Sigmoid(),
    Linear(512, 10),
    batch_size=250,
)
net.configure(learning_rate=0.001, inference_rate=0.08, optimizer="ADAM")

X = np.array([-1, -1, -1, 1, 1, -1, 1, 1], dtype=np.float32)
Y = np.array([-1, 1, 1, -1], dtype=np.float32)

# One call settles, updates weights, and reports energy.
for epoch in range(1500):
  energy = net.train_step_with_projection(X, Y, 150)

predictions = net.predict_with_projection(X, 150)
`;

const cppcode = `#include <deepity/networks/SimplePCNetwork.h>
#include <random>

Deep::SimplePCNetwork net(4);
net.AddLayer(784, 512, 0.001f, 0.08f, 0.0f, Deep::ActivationType::SIGMOID, Deep::ActivationType::dSIGMOID);
net.AddLayer(512, 512, 0.001f, 0.08f, 0.0f, Deep::ActivationType::SIGMOID, Deep::ActivationType::dSIGMOID);
net.AddLayer(512, 10, 0.001f, 0.08f, 0.0f, Deep::ActivationType::LINEAR, Deep::ActivationType::dLINEAR);
net.AddLayer(10, 0, 0.001f, 0.08f, 0.0f, Deep::ActivationType::LINEAR, Deep::ActivationType::dLINEAR);

net.SetOptimizer(Deep::OptimizerType::ADAM);
net.Compile();

std::mt19937 rng(42);
net.RandomizeWeights(rng);

std::vector<float> X = {-1, -1, -1, 1, 1, -1, 1, 1};
std::vector<float> Y = {-1, 1, 1, -1};

// One call settles, updates weights, and reports energy.
for (int epoch = 0; epoch < 1500; ++epoch)
    float energy = net.TrainStepWithProjection(X, Y, 150);

std::vector<float> predictions = net.PredictWithProjection(X, 150);`;

const heroStyle = {
  backgroundImage: `url(${import.meta.env.BASE_URL}flowerpot.webp)`,
  backgroundSize: "cover",
  backgroundPosition: "center 65%",
};

const syntaxCustomStyle = {
  margin: 0,
  padding: "1.25rem",
  fontSize: "14px",
  fontFamily: "Fira Code, monospace",
  background: "#fafafa",
};

export default function HomePage() {
  return (
    <div className="bg-[#e4e6e7]">
      <div
        className="min-h-screen flex flex-col items-center justify-center AllianceNo2 gap-4 text-center border-b border-[#202d3b]-200 backdrop-blur-md"
        style={heroStyle}
      >
        <span className="text-4xl AllianceNo2">Welcome to Deepity</span>
        <span className="text-lg AllianceNo2 mb-10">
          A high-performance implementation of
          <br /> Predictive Coding Networks in C++ and Python.
        </span>
        <a
          href="#Explained"
          className="group flex items-center gap-4 shadow-lg border border-black px-5 py-3 mt-3 text-base text-black font-bold no-underline transition-colors duration-300 hover:bg-black hover:text-white hover:scale-105"
        >
          Get Started
          <Arrow />
        </a>
      </div>

      <section className="bg-[#e6e8e9] border-b border-[#202d3b]-200 px-8 py-20">
        <div className="mx-auto max-w-5xl">
          <h2 className="text-3xl AllianceNo2">Proven Performance</h2>

          <p className="mt-3 max-w-2xl text-base text-black/70 AllianceNo2 text-lg">
            Deepity's DKPPCN reaches 97.73% test accuracy on MNIST, approaching
            PyTorch's feedforward backprop accuracy while training entirely on
            the CPU.
          </p>
          <Suspense fallback={<div className="h-96" />}>
            <BenchmarkChart />
          </Suspense>
        </div>
      </section>

      <section
        className="bg-[#e6e8e9] border-b border-[#202d3b]-200 px-8 py-20"
        id="Explained"
      >
        <div className="mx-auto max-w-5xl justify-center align-items text-center">
          <h2 className="text-3xl AllianceNo2">What is Predictive Coding?</h2>

          <p className="mt-3 text-black text-lg AllianceNo2">
            Deepity implements Predictive Coding Networks, where neurons
            iteratively minimize local prediction errors rather than propagating
            gradients backward through the entire network.
          </p>
          <Suspense fallback={<div className="h-48" />}>
            <PCNRepresentation />
          </Suspense>
          <p className="mt-3 text-black text-lg AllianceNo2">
            This approach is inspired by the brain's predictive coding theory,
            which suggests that the brain constantly generates predictions about
            incoming sensory information and updates its internal model based on
            the prediction errors. This architecture is capable of learning
            complex representations, continuous learning, and even generalized
            learning across tasks.
            <br />
            <br />
            The problem is this algorithm is computationally expensive and slow
            to train, which is why Deepity focuses on optimizing its
            implementation for better performance.
          </p>
        </div>
      </section>

      <section className="bg-[#e6e8e9] border-b border-[#202d3b]-200 px-8 py-20">
        <div className="mx-auto max-w-5xl">
          <h2 className="text-3xl AllianceNo2 justify-center text-center">
            Why Deepity?
          </h2>
          <ul className="mt-5 list-disc list-inside text-black/70 text-lg AllianceNo2">
            <li className="m-3">
              <b>CPU-First</b>: Bundled with OpenBLAS and OpenMP for optimized
              CPU performance, making it ideal for edge devices and low-power
              environments. Custom SIMD kernels for Intel and AMD CPUs, with ARM
              support in progress.
            </li>
            <li className="m-3">
              <b>Strongly Bound</b>: Built with nanobind for seamless
              integration with Python ecosystems and a wheel size under 1MB.
            </li>
            <li className="m-3">
              <b>Well-Documented</b>: Comprehensive documentation (using
              Doxygen) and examples to help users get started quickly. Examples
              are tested during every push with GitHub Actions.
            </li>
            <li className="m-3">
              <b>Open-Source</b>: Licensed under the MIT License, making it
              freely available for anyone to use and contribute to. Deepity was
              made by a single developer and is actively maintained with a focus
              on performance and usability. For my full story, check out{" "}
              <a
                href="https://ra4ster.github.io"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:underline text-blue-600"
              >
                my website here
              </a>
              .
            </li>
          </ul>
        </div>
      </section>

      <section className="bg-[#e6e8e9] border-b border-[#202d3b]-200 px-8 py-10">
        <div className="mx-auto max-w-5xl">
          <h2 className="text-3xl AllianceNo2">Code Examples</h2>
          <p className="mt-2 text-black/70 AllianceNo2">
            Deepity is designed to be easy to use and integrate into existing
            projects. Here are some code examples to get you started.
          </p>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <CodeBlock
              language="python"
              code={pythoncode}
              icon={PythonIcon}
              title="Python"
            />

            <CodeBlock
              language="cpp"
              code={cppcode}
              icon={CppIcon}
              title="C++"
            />
          </div>
        </div>
      </section>

      <section className="bg-[#e6e8e9] border-b border-[#202d3b]/20 px-8 py-20">
        <div className="mx-auto max-w-5xl flex flex-col md:flex-row items-start md:items-center justify-between gap-12">
          <div className="flex-1">
            <h2 className="text-3xl AllianceNo2">Interested in Development?</h2>

            <p className="mt-3 max-w-2xl text-black/70 AllianceNo1">
              Deepity is an open-source implementation of Predictive Coding
              Networks, and I welcome contributions from the community. If
              you're interested in contributing to the project, please check out
              the GitHub repository and feel free to submit pull requests or
              open issues. Your contributions can help improve the performance,
              usability, and documentation of Deepity.
            </p>

            <div className="mt-8 flex flex-wrap gap-4">
              <a
                href="mailto:jackrose2335@gmail.com"
                target="_blank"
                rel="noopener noreferrer"
                className="border border-black hover:shadow-lg px-5 py-3 font-bold no-underline text-black transition-colors hover:bg-black hover:text-white AllianceNo1"
              >
                Contact Me
              </a>

              <Link
                to="/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="border border-black hover:shadow-lg px-5 py-3 font-bold no-underline text-black transition-colors hover:bg-black hover:text-white AllianceNo1"
              >
                Learn More Theory
              </Link>
            </div>
          </div>

          <iframe
            src="https://discord.com/widget?id=1545278281807433808&theme=dark"
            width="350"
            height="350"
            allowtransparency="true"
            sandbox="allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts"
            className="shrink-0 w-full md:w-[350px] shadow-xl hover:shadow-2xl hover:scale-102 transition-all duration-300"
          />
        </div>
      </section>
      <MainFooter />
    </div>
  );
}
