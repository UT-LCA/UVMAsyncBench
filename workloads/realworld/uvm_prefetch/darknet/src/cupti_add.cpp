#include "cupti_add.h"

static void
printActivity(CUpti_Activity *record)
{
    switch (record->kind)
    {
    case CUPTI_ACTIVITY_KIND_KERNEL:
    {
        int status;
        CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
        printf("CUPTI,%s,%llu,%llu,%llu\n",
               abi::__cxa_demangle(kernel->name, 0, 0, &status),
               (unsigned long long)(kernel->start),
               (unsigned long long)(kernel->end),
               (unsigned long long)(kernel->end - startTimestamp) - (kernel->start - startTimestamp));
        break;
    }
    case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
        CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
        const char *callback_name;
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &callback_name);
        printf("RUNTIME %s (cbid=%u) [ %llu - %llu ] process %u, thread %u, correlation %u\n",
               callback_name, api->cbid,
               (unsigned long long)(api->start - startTimestamp),
               (unsigned long long)(api->end - startTimestamp),
               api->processId, api->threadId, api->correlationId);
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

void initTrace()
{
    size_t attrValue = 0, attrValueSize = sizeof(size_t);

    // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

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