import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface Testimonial {
  text: string;
  author: string;
  position: string;
  company: string;
  logo: string;
}

const testimonials: Testimonial[] = [
  {
    text: "Ricco.AI transformed our business operations. Their AI integration solutions helped us achieve 40% more efficiency in our customer service.",
    author: "Sarah Chen",
    position: "CEO",
    company: "TechFlow Solutions",
    logo: "https://images.unsplash.com/photo-1560179707-f14e90ef3623?auto=format&fit=crop&q=80&w=100&h=100"
  },
  {
    text: "The AI team building service has been a game-changer. We've automated complex processes and seen remarkable improvements in our productivity.",
    author: "Michael Rodriguez",
    position: "Operations Director",
    company: "InnovateNow Inc",
    logo: "https://images.unsplash.com/photo-1599305445671-ac291c95aaa9?auto=format&fit=crop&q=80&w=100&h=100"
  },
  {
    text: "Their business analytics solutions provided insights that helped us make data-driven decisions and grow our revenue by 25% in just six months.",
    author: "Emily Watson",
    position: "Marketing Manager",
    company: "Growth Dynamics",
    logo: "https://images.unsplash.com/photo-1572021335469-31706a17aaef?auto=format&fit=crop&q=80&w=100&h=100"
  }
];

const TestimonialCarousel: React.FC = () => {
  const [current, setCurrent] = useState(0);
  const [direction, setDirection] = useState(0);

  const slideVariants = {
    enter: (direction: number) => ({
      x: direction > 0 ? 1000 : -1000,
      opacity: 0
    }),
    center: {
      zIndex: 1,
      x: 0,
      opacity: 1
    },
    exit: (direction: number) => ({
      zIndex: 0,
      x: direction < 0 ? 1000 : -1000,
      opacity: 0
    })
  };

  const swipeConfidenceThreshold = 10000;
  const swipePower = (offset: number, velocity: number) => {
    return Math.abs(offset) * velocity;
  };

  const paginate = (newDirection: number) => {
    setDirection(newDirection);
    setCurrent((current + newDirection + testimonials.length) % testimonials.length);
  };

  useEffect(() => {
    const timer = setInterval(() => {
      paginate(1);
    }, 5000);
    return () => clearInterval(timer);
  }, [current]);

  return (
    <div className="relative w-full max-w-4xl mx-auto px-4">
      <div className="relative h-80 overflow-hidden">
        <div className="absolute top-1/3 -left-4 -right-4 flex items-center justify-between z-10 px-4">
          <button
            className="text-primary/60 hover:text-primary transition-colors duration-300"
            onClick={() => paginate(-1)}
          >
            <ChevronLeft className="w-8 h-8" />
          </button>

          <button
            className="text-primary/60 hover:text-primary transition-colors duration-300"
            onClick={() => paginate(1)}
          >
            <ChevronRight className="w-8 h-8" />
          </button>
        </div>

        <AnimatePresence initial={false} custom={direction}>
          <motion.div
            key={current}
            custom={direction}
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{
              x: { type: "spring", stiffness: 300, damping: 30 },
              opacity: { duration: 0.2 }
            }}
            drag="x"
            dragConstraints={{ left: 0, right: 0 }}
            dragElastic={1}
            onDragEnd={(e, { offset, velocity }) => {
              const swipe = swipePower(offset.x, velocity.x);
              if (swipe < -swipeConfidenceThreshold) {
                paginate(1);
              } else if (swipe > swipeConfidenceThreshold) {
                paginate(-1);
              }
            }}
            className="absolute w-full"
          >
            <div className="bg-dark-lighter p-8 rounded-lg shadow-lg">
              <div className="flex items-center mb-6">
                <img
                  src={testimonials[current].logo}
                  alt={testimonials[current].company}
                  className="w-16 h-16 rounded-full object-cover mr-4"
                />
                <div>
                  <h4 className="text-lg font-semibold">{testimonials[current].company}</h4>
                  <p className="text-gray-400">{testimonials[current].author}, {testimonials[current].position}</p>
                </div>
              </div>
              <p className="text-lg text-gray-300 italic mb-6">"{testimonials[current].text}"</p>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="flex justify-center mt-4 space-x-2">
        {testimonials.map((_, index) => (
          <button
            key={index}
            className={`w-2 h-2 rounded-full transition-colors duration-300 ${
              index === current ? 'bg-primary' : 'bg-gray-600'
            }`}
            onClick={() => setCurrent(index)}
          />
        ))}
      </div>
    </div>
  );
};

export default TestimonialCarousel;