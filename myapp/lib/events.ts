// Simple global event emitter for SSE
type Listener = (data: any) => void;

class EventEmitter {
  private listeners: Set<Listener> = new Set();

  subscribe(listener: Listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  emit(data: any) {
    this.listeners.forEach((l) => l(data));
  }
}

// Ensure the event bus is a singleton even across Hot Module Replacement in Next.js
const globalForEvents = globalThis as unknown as {
  eventBus: EventEmitter | undefined;
};

export const eventBus = globalForEvents.eventBus ?? new EventEmitter();

if (process.env.NODE_ENV !== "production") globalForEvents.eventBus = eventBus;
