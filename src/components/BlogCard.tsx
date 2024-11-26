import React from 'react';
import { motion } from 'framer-motion';

interface BlogCardProps {
  title: string;
  excerpt: string;
  date: string;
  readTime: string;
}

const BlogCard: React.FC<BlogCardProps> = ({ title, excerpt, date, readTime }) => {
  return (
    <motion.article
      className="bg-dark-lighter rounded-lg overflow-hidden border-2 border-blue-500"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      whileHover={{ y: -8 }}
      transition={{ duration: 0.3 }}
    >
      <div className="p-6">
        <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
          <time>{new Date(date).toLocaleDateString()}</time>
          <span>{readTime}</span>
        </div>
        <h3 className="text-xl font-bold mb-2 hover-glow">{title}</h3>
        <p className="text-gray-400 text-sm mb-4">{excerpt}</p>
        <motion.button
          className="text-primary hover:text-primary-dark transition-colors duration-300"
          whileHover={{ x: 5 }}
          transition={{ duration: 0.2 }}
        >
          Read More →
        </motion.button>
      </div>
    </motion.article>
  );
};

export default BlogCard;