import React, { useRef, useState } from 'react'

const MusicButton = () => {
  const [play, setPlay] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  const togglePlay = () => {
    if (play) {
      audioRef.current?.pause();
      setPlay(false);
    } else {
      audioRef.current?.play();
      setPlay(true);
    }
  };

  return (
    <>
      <button onClick={togglePlay}>{play ? 'Pause' : 'Play'}</button>
      <audio
        ref={audioRef}
        src="/audio/generated.mid"
        onEnded={() => setPlay(false)}
        onPlay={() => setPlay(true)}
        onPause={() => setPlay(false)}
      />
    </>
  )
}

export default MusicButton