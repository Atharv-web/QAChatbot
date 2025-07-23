import { motion } from 'framer-motion';

export default function GlassCard({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="bg-white/5 backdrop-blur-lg border border-white/20 shadow-2xl rounded-2xl p-5 w-full max-w-2xl min-h flex flex-col overflow "
      whileHover={{ scale: 1.02 }}
    >
      {children}
    </motion.div>
  );
}