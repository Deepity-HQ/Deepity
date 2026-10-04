import Navbar from "./components/Navbar";
import HomePage from "./HomePage";
import NotFound from "./NotFound";
import "./index.css";

import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import TutorialLanding from "./TutorialLanding";
import Tutorial0_Intro from "./components/tutorials/Tutorial0_Intro";
import Tutorial1_Beginning from "./components/tutorials/Tutorial1_Beginning";
import SplashScreen from "./components/SplashScreen";

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

export default function App() {
  const [showSplash, setShowSplash] = useState(true);

  return (
    <>
      {showSplash && <SplashScreen onDone={() => setShowSplash(false)} />}

      <div style={{ position: "relative", zIndex: 1 }}>
        <BrowserRouter basename="/Deepity">
          <ScrollToTop />
          <Navbar />

          <main>
            <Routes>
              <Route path="" element={<HomePage />} />
              <Route path="/tutorial" element={<TutorialLanding />} />
              <Route
                path="/tutorial/0-introduction"
                element={<Tutorial0_Intro />}
              />
              <Route
                path="/tutorial/1-beginning"
                element={<Tutorial1_Beginning />}
              />
              <Route path="/notfound" element={<NotFound />} />
            </Routes>
          </main>
        </BrowserRouter>
      </div>
    </>
  );
}
