import React, { useRef, useState } from 'react'

const MusicButton = () => {
  const [play, setPlay] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  const togglePlay = () => {
    if (play) {
      audioRef.current?.pause();
    } else {
      audioRef.current?.play();
    }

  };

  return (
    <>
      <button onClick={togglePlay}>{play ? 'Pause' : 'Play'}</button>
      <audio
        ref={audioRef}
        src=".../audio/generated.mid"
        onEnded={() => setPlay(false)}
      />
    </>
  )
}

export default MusicButton