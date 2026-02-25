import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import MusicButton from "./MusicButton";

describe("MusicButton", () => {
  let playSpy: any;
  let pauseSpy: any;

  beforeEach(() => {
    playSpy = vi.spyOn(window.HTMLMediaElement.prototype, "play").mockImplementation(() => Promise.resolve());
    pauseSpy = vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(() => { });
  });

  afterEach(() => {
    playSpy.mockRestore();
    pauseSpy.mockRestore();
  });

  it("calls play when button is clicked", async () => {
    const user = userEvent.setup();

    render(<MusicButton />);
    const button = screen.getByRole("button");

    await user.click(button);

    expect(playSpy).toHaveBeenCalled();
    expect(button.textContent).toBe("Pause");
  });

  it("calls pause when button is clicked twice", async () => {
    const user = userEvent.setup();

    render(<MusicButton />);
    const button = screen.getByRole("button");

    await user.click(button);
    await user.click(button);

    expect(playSpy).toHaveBeenCalled();
    expect(pauseSpy).toHaveBeenCalled();
    expect(button.textContent).toBe("Play");
  });
});
