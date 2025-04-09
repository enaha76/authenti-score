"use client";

import { motion } from "framer-motion";
import { PropsWithChildren } from "react";

export function MotionWrapper({ children, ...props }: PropsWithChildren<any>) {
  return <motion.div {...props}>{children}</motion.div>;
}