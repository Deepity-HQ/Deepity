import React from "react";
import MainFooter from "../MainFooter";
import UtilButton from "../UtilButton";
import CodeBlock from "../CodeBlock";
import CmdLogo from "../../assets/cmd.png";
import buildingImage from "../../assets/buildingdeepity.png";

const buildCode = `# Clone the Deepity repository
git clone https://github.com/Ra4ster/Deepity.git
cd Deepity

# Compile the C++ backend
python build.py`;

const configCode = `# Configure backend and hyperparameters
model.configure(
    learning_rate=0.001,
    inference_rate=0.05,
    optimizer="adam",
    regularization=1e-4
)`;

const trainCode = `# Train the network automatically
model.fit(x_train, y_train, epochs=10, batch_size=64)

# Run inference steps for local energy minimization
predictions = model.predict(x_test, settling_steps=20)`;

export default function Tutorial1_Beginning() {
  return (
    <div className="bg-gradient-to-t from-[#DDDDDD] to-[#e4e6e7] min-h-screen pt-[90px]">
      <div className="min-h-screen max-w-7xl mx-auto pb-12">
        <UtilButton
          backLink="/tutorial/0-intro"
          nextLink="/tutorial/2-architecture"
        />

        <div className="px-8 my-5">
          <h1 className="AllianceNo2 text-4xl font-bold m-5">
            Tutorial 1: Understanding the Deepity Library
          </h1>
          <p className="text-lg m-5 AllianceNo1 max-w-4xl text-gray-800">
            Now that you understand the theory behind Predictive Coding
            Networks, it is time to look at Deepity itself. We will cover the
            library's design, how to compile its C++ backend, and the core
            concepts required to construct a SimplePCN to classify handwritten
            digits.
          </p>
        </div>

        <div className="border-t border-gray-400 my-8 mx-8" />

        <div className="px-8 md:px-12 space-y-10 AllianceNo1 text-lg text-gray-800">
          <section>
            <h2 className="text-2xl font-bold mb-4 AllianceNo2">
              1. What Makes Deepity Special?
            </h2>
            <p className="mb-4">
              The historic bottleneck for Predictive Coding Networks (PCNs) is
              speed. Standard feedforward neural networks rely on standard
              matrix multiplications that modern GPUs compute quickly. PCNs,
              however, require iterative settling loops to simulate local energy
              minimization.
            </p>
            <p className="mb-6">
              Deepity addresses this by implementing a C++ backend with a Python
              interface. It uses fused SIMD loops on the CPU and custom CUDA
              kernels to manage memory operations efficiently. This allows the
              library's PCNs to train at speeds comparable to standard
              backpropagation models.
            </p>
            <div className="mx-auto flex w-full flex-col items-center md:max-w-[500px]">
              <img
                src="../nanobind.jpg"
                alt="Feedforward neural networks visualized"
                className="block w-full rounded shadow-md border border-black/15"
              />
              <p className="mt-3 text-center text-sm text-gray-600 AllianceNo1">
                <a
                  href="https://github.com/wjakob/nanobind"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  Fig. 1: Nanobind, Deepity's lightweight Python binding
                  dependency.
                </a>
              </p>
            </div>
          </section>

          <div className="border-t border-gray-400 my-8 mx-8" />

          <section>
            <h2 className="text-2xl font-bold mb-4 AllianceNo2">
              2. Building, Loading, and GitHub Workflows
            </h2>
            <p className="mb-4">
              Because Deepity relies on a C++ core, the backend must be compiled
              for your specific hardware before use. The repository includes a
              build script to handle this compilation. Once built, the module
              can be imported directly into your Python environment.
            </p>
            <p className="mb-6">
              The repository also includes GitHub workflows that automate
              cross-platform building and testing upon every push to maintain
              stability across different environments.
            </p>

            <div className="mx-auto flex w-full flex-col items-center md:max-w-[1000px] mb-8">
              <img
                src={buildingImage}
                alt="Building Deepity"
                className="block h-auto w-full rounded shadow-md border border-black/15"
                height="717"
                width="2586"
              />
              <p className="mt-3 text-center text-sm text-gray-600 AllianceNo1">
                Fig. 2: Building Deepity using the provided{" "}
                <code>build.py</code> script.
              </p>
            </div>

            <div className="mx-auto w-full max-w-[800px] flex flex-col items-center">
              <CodeBlock
                title="Terminal"
                icon={CmdLogo}
                language="bash"
                code={buildCode}
              />
              <p className="mt-3 text-center text-sm text-gray-600 AllianceNo1">
                Fig. 3: Compiling the Deepity backend via the command line.
              </p>
            </div>
          </section>

          <div className="border-t border-gray-400 my-8 mx-8" />

          <section>
            <h2 className="text-2xl font-bold mb-4 AllianceNo2">
              3. Declarative Architecture
            </h2>
            <p className="mb-4">
              Networks in Deepity are initialized declaratively. Rather than
              appending layers sequentially, you define the complete structure
              upfront by passing layer components directly into the network's
              constructor.
            </p>
            <p className="mb-6">
              For instance, a simple network is constructed by providing
              dimension mappings and activation functions at initialization. The
              library will then verify that the layer dimensions match from the
              input layer through to the output layer before proceeding.
            </p>
          </section>

          <div className="border-t border-gray-400 my-8 mx-8" />

          <section>
            <h2 className="text-2xl font-bold mb-4 AllianceNo2">
              4. Configuring the Backend
            </h2>
            <p className="mb-4">
              After defining the architecture in Python, the network must be
              configured. This step initializes the corresponding C++ backend
              structures and allocates the necessary memory.
            </p>
            <p className="mb-6">
              During this step, you set the hyperparameters for optimization and
              inference. This includes the weight-learning rate, settling
              inference rate, regularization parameters, and the chosen
              optimizer.
            </p>

            <div className="mx-auto w-full max-w-[800px] flex flex-col items-center mt-6">
              <CodeBlock title="Python" language="python" code={configCode} />
              <p className="mt-3 text-center text-sm text-gray-600 AllianceNo1">
                Fig. 4: Configuring the optimization and inference parameters.
              </p>
            </div>
          </section>

          <div className="border-t border-gray-400 my-8 mx-8" />

          <section>
            <h2 className="text-2xl font-bold mb-4 AllianceNo2">
              5. Training and Inference
            </h2>
            <p className="mb-4">
              Once configured, the network is ready for training. Deepity
              provides built-in <code>fit</code> functions for standard
              training, or you can write manual training loops by executing
              single steps on data batches.
            </p>
            <p className="mb-6">
              Because predictive coding relies on state minimization, generating
              predictions requires running the network through iterative
              inference steps to let the local energy settle. Once settled, the
              final layer's state represents the network's output, which can
              then be evaluated against your target classes.
            </p>

            <div className="mx-auto w-full max-w-[800px] flex flex-col items-center mt-6">
              <CodeBlock title="Python" language="python" code={trainCode} />
              <p className="mt-3 text-center text-sm text-gray-600 AllianceNo1">
                Fig. 5: Executing training and settled inference.
              </p>
            </div>
          </section>
        </div>
      </div>
      <MainFooter />
    </div>
  );
}
