#include "cupti_add.h"

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
    switch (kind)
    {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
        return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
        return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
        return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
        return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
        return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
        return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
        return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
        return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
        return "HtoH";
    default:
        break;
    }

    return "<unknown>";
}

static const char *
getUvmCounterKindString(CUpti_ActivityUnifiedMemoryCounterKind kind)
{
    switch (kind)
    {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
        return "BYTES_TRANSFER_HTOD";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
        return "BYTES_TRANSFER_DTOH";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
        return "CPU_PAGE_FAULTS";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
        return "GPU_PAGE_FAULTS";
    default:
        break;
    }
    return "<unknown>";
}

static void
printActivity(CUpti_Activity *record)
{
    switch (record->kind)
    {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        {
            int status;
            CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
            printf("KERNEL %s, %llu, %llu, %llu\n",
                abi::__cxa_demangle(kernel->name, 0, 0, &status),
                (unsigned long long)(kernel->start),
                (unsigned long long)(kernel->end),
                (unsigned long long)(kernel->end) - (kernel->start));
            break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        {
            CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
            const char *callback_name;
            cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &callback_name);
            // printf("RUNTIME %s (cbid=%u) [ %llu - %llu ] process %u, thread %u, correlation %u\n",
            //        callback_name, api->cbid,
            //        (unsigned long long)(api->start - startTimestamp),
            //        (unsigned long long)(api->end - startTimestamp),
            //        api->processId, api->threadId, api->correlationId);
            printf("RUNTIME %s (cbid=%u), %llu,%llu,%llu, process %u, thread %u, correlation %u\n",
                callback_name, api->cbid,
                (unsigned long long)(api->start),
                (unsigned long long)(api->end),
                (unsigned long long)(api->end - api->start),
                api->processId, api->threadId, api->correlationId);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            CUpti_ActivityMemcpy4 *memcpy = (CUpti_ActivityMemcpy4 *)record;
            printf("MEMCPY %s, size %llu, %llu, %llu, %llu\n",
                   getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
                   (unsigned long long)memcpy->bytes,
                   (unsigned long long)(memcpy->start),
                   (unsigned long long)(memcpy->end),
                   (unsigned long long)(memcpy->end) - (memcpy->start));
            break;
        }
        case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
        {
            CUpti_ActivityUnifiedMemoryCounter2 *uvm = (CUpti_ActivityUnifiedMemoryCounter2 *)record;
            printf("UVM MEMCPY %s, size %llu, %llu, %llu, %llu \n",
                   getUvmCounterKindString(uvm->counterKind),
                   (unsigned long long)uvm->value,
                   (unsigned long long)(uvm->start),
                   (unsigned long long)(uvm->end),
                   (unsigned long long)(uvm->end - uvm->start));
            break;
        }
    }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL)
    {
        printf("Error: out of memory\n");
        exit(-1);
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;
    if (validSize > 0)
    {
        do
        {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS)
            {
                printActivity(record);
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else
            {
                CUPTI_CALL(status);
            }
        } while (1);

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0)
        {
            printf("Dropped %u activity records\n", (unsigned int)dropped);
        }
    }

    free(buffer);
}

// void initTrace() {
//     return;
// }

// void finiTrace() {
//     return;
// }


void initTrace()
{
    size_t attrValue = 0, attrValueSize = sizeof(size_t);

    CUpti_ActivityUnifiedMemoryCounterConfig config[2];

    // configure unified memory counters
    config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
    config[0].deviceId = 0;
    config[0].enable = 1;

    config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
    config[1].deviceId = 0;
    config[1].enable = 1;

    CUptiResult res = cuptiActivityConfigureUnifiedMemoryCounter(config, 2);
    if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED)
    {
        printf("Test is waived, unified memory is not supported on the underlying platform.\n");
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE)
    {
        printf("Test is waived, unified memory is not supported on the device.\n");
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES)
    {
        printf("Test is waived, unified memory is not supported on the non-P2P multi-gpu setup.\n");
    }
    else
    {
        CUPTI_CALL(res);
    }

    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
    // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_COUNT));

    // CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT 
    // CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT

    // Register callbacks for buffer requests and for buffers completed by CUPTI.
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    // Optionally get and set activity attributes.
    // Attributes can be set by the CUPTI client to change behavior of the activity API.
    // Some attributes require to be set before any CUDA context is created to be effective,
    // e.g. to be applied to all device buffer allocations (see documentation).
    CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
    printf("%s = %llu B\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);
    attrValue *= 2;
    CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

    CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
    printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);
    attrValue *= 2;
    CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));

    CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

void finiTrace()
{
    // Force flush any remaining activity buffers before termination of the application
    CUPTI_CALL(cuptiActivityFlushAll(1));
}

