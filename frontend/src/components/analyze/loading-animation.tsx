"use client";

import React from 'react';
import LottiePlayer from 'react-lottie-player';

// Move animation data outside the component
const animationData = {
  v: "5.5.7",
  meta: {
    g: "LottieFiles AE 0.1.20",
    a: "",
    k: "",
    d: "",
    tc: ""
  },
  fr: 30,
  ip: 0,
  op: 60,
  w: 400,
  h: 400,
  nm: "Loading",
  ddd: 0,
  assets: [],
  layers: [
    {
      ddd: 0,
      ind: 1,
      ty: 4,
      nm: "circle 3",
      sr: 1,
      ks: {
        o: {
          a: 1,
          k: [
            { i: { x: [0.667], y: [1] }, o: { x: [0.333], y: [0] }, t: 20, s: [100] },
            { i: { x: [0.667], y: [1] }, o: { x: [0.333], y: [0] }, t: 40, s: [30] },
            { t: 60, s: [100] }
          ],
          ix: 11
        },
        r: { a: 0, k: 0, ix: 10 },
        p: { a: 0, k: [270, 200, 0], ix: 2 },
        a: { a: 0, k: [0, 0, 0], ix: 1 },
        s: { a: 0, k: [100, 100, 100], ix: 6 }
      },
      ao: 0,
      shapes: [{
        ty: "gr",
        it: [
          {
            d: 1,
            ty: "el",
            s: { a: 0, k: [40, 40], ix: 2 },
            p: { a: 0, k: [0, 0], ix: 3 },
            nm: "Ellipse Path 1",
            mn: "ADBE Vector Shape - Ellipse",
            hd: false
          },
          {
            ty: "fl",
            c: { a: 0, k: [0.337, 0.471, 0.894, 1], ix: 4 },
            o: { a: 0, k: 100, ix: 5 },
            r: 1,
            bm: 0,
            nm: "Fill 1",
            mn: "ADBE Vector Graphic - Fill",
            hd: false
          },
          {
            ty: "tr",
            p: { a: 0, k: [0, 0], ix: 2 },
            a: { a: 0, k: [0, 0], ix: 1 },
            s: { a: 0, k: [100, 100], ix: 3 },
            r: { a: 0, k: 0, ix: 6 },
            o: { a: 0, k: 100, ix: 7 },
            sk: { a: 0, k: 0, ix: 4 },
            sa: { a: 0, k: 0, ix: 5 },
            nm: "Transform"
          }
        ],
        nm: "Ellipse 1",
        np: 3,
        cix: 2,
        bm: 0,
        ix: 1,
        mn: "ADBE Vector Group",
        hd: false
      }],
      ip: 0,
      op: 900,
      st: 0,
      bm: 0
    }
  ],
  markers: []
};

export function LoadingAnimation() {
  return (
    <div className="flex flex-col items-center">
      <LottiePlayer
        autoplay
        loop
        src={animationData}
        style={{ height: '120px', width: '120px' }}
      />
      <p className="text-gray-600 mt-4 animate-pulse">Analyse en cours...</p>
    </div>
  );
}
