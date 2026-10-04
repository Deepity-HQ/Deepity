import { useState } from "react";
import MainFooter from "./components/MainFooter";
import TutorialCard from "./components/TutorialCard";

const tutorialsData = [
  {
    title: "0. Intro to Predictive Coding",
    subtitle: "Beginner",
    description:
      "For people with little to no knowledge of predictive coding and Deepity.",
    img: "./doorway.webp",
    link: "0-introduction",
  },
  {
    title: "1. Understanding the Deepity Library",
    subtitle: "Beginner",
    description:
      "Explanation of what makes Deepity special and how to load/use it.",
    img: "./laptop.webp",
    link: "/tutorial/1-beginning",
  },
  {
    title: "2. Training a SimplePCN on XOR",
    subtitle: "Intermediate",
    description:
      "Get your first experience training the canonical PCN on a simple function.",
    img: "./ripple.webp",
    link: "/notfound",
  },
  {
    title: "3. Adding Helpful Hyperparameters",
    subtitle: "Intermediate",
    description:
      "Now that you know how to use a SimplePCN, try converting to a SequentialPCN and tweaking parameters like precision and lambda.",
    img: "./dartboard.webp",
    link: "/notfound",
  },
  {
    title: "4. Running a Convolutional PCN",
    subtitle: "Hard",
    description:
      "Attempt to convolve the output over an image in 2D, capturing higher accuracy.",
    img: "./dimension.webp",
    link: "/notfound",
  },
  {
    title: "5. Direct Kolen-Pollack Networks on MNIST",
    subtitle: "Hard",
    description:
      "Research how propagating feedback can make convergence hyperfast.",
    img: "./wired.webp",
    link: "/notfound",
  },
];

export default function TutorialLanding() {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredTutorials = tutorialsData.filter(
    (tutorial) =>
      tutorial.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      tutorial.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      tutorial.subtitle.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  return (
    <div className="bg-gradient-to-t from-[#DDDDDD] to-[#e4e6e7] min-h-screen pt-[90px]">
      <div className="min-h-screen">
        <div className="max-w-7xl mx-auto px-4 pb-2 pt-4">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 flex items-center pl-4 pointer-events-none">
              <svg
                className="w-5 h-5 text-gray-400"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 20 20"
              >
                <path
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"
                />
              </svg>
            </div>
            <input
              type="text"
              placeholder="Search tutorials..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full p-4 pl-12 text-gray-800 bg-white rounded-2xl shadow-sm hover:shadow-md focus:shadow-md focus:outline-none transition-shadow border-none"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 p-4 max-w-7xl mx-auto w-full">
          {filteredTutorials.length > 0 ? (
            filteredTutorials.map((tutorial, index) => (
              <TutorialCard
                key={index}
                title={tutorial.title}
                subtitle={tutorial.subtitle}
                description={tutorial.description}
                img={tutorial.img}
                link={tutorial.link}
              />
            ))
          ) : (
            <div className="col-span-full text-center py-10 text-gray-500">
              No tutorials found matching "{searchQuery}"
            </div>
          )}
        </div>
      </div>
      <MainFooter />
    </div>
  );
}
