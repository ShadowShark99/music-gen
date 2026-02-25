// App.test.jsx

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Title from "./components/Title";

describe("App component", () => {
  it("renders correct heading", () => {
    render(<Title title="tess" />);
    // using regex with the i flag allows simpler case-insensitive comparison
    expect(screen.getByRole("heading").textContent).toMatch(/tess/i);
  });
});